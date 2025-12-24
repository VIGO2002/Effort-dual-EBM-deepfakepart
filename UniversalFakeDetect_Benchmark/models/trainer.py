import functools
import torch
import torch.nn as nn
from .base_model import BaseModel, init_weights
import sys
from models import get_model
from transformers import get_cosine_schedule_with_warmup

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt.arch, opt)
        self.lr = opt.lr
        
        # EBM_Head 初始化
        if hasattr(self.model, 'fc'):
            for m in self.model.fc.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0.0, opt.init_gain)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)

        # 参数冻结逻辑
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if 'fc.' in name: 
                    params.append(p) 
                    p.requires_grad = True
                elif any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                    params.append(p)
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            print(f">>> Backbone fixed. Training {len(params)} tensors (Head + SVD Residuals).")
        else:
            print("Your backbone is not fixed. Training all parameters.")
            params = self.model.parameters()

        # 优化器
        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        # 【核心修改】损失函数升级
        # 1. CE Loss 用于基础分类
        self.loss_fn = nn.CrossEntropyLoss()
        # 2. Margin Loss 用于强化 EBM 边界 (可选，如果你想追求极致)
        self.margin = 0.5 
        self.lambda_ebm = 0.5 

        self.model.to(opt.gpu_ids[0])
        
        # Warmup Scheduler
        self.scheduler = None
        if hasattr(opt, 'warmup_steps') and opt.warmup_steps > 0:
            print(f">>> Using Cosine Scheduler with {opt.warmup_steps} warmup steps.")
            try:
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=opt.warmup_steps, num_training_steps=opt.niter * 1000 
                )
            except:
                pass 

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).long() 

    def forward(self):
        # Dual-EBM 输出 shape 为 [Batch, 2] -> [-E_real, -E_fake]
        self.output = self.model(self.input)
        if self.output.dim() == 1:
             pass 

    def get_loss(self):
        # --- 纯正 Dual-Head EBM 优化策略 ---
        # self.output = [-E_real, -E_fake] (即 Logits)
        
        # 1. 基础 CrossEntropy (保证收敛)
        ce_loss = self.loss_fn(self.output, self.label)
        
        # 2. EBM Margin Loss (增强泛化)
        # 逻辑：
        # 对于真图 (Label=0): 我们希望 -E_real >> -E_fake (即 E_real << E_fake)
        # 对于假图 (Label=1): 我们希望 -E_fake >> -E_real (即 E_fake << E_real)
        
        logits_real = self.output[:, 0]
        logits_fake = self.output[:, 1]
        
        # diff = Logit_Real - Logit_Fake
        # 如果是真图，diff 应该很大；假图，diff 应该很小(负数)
        diff = logits_real - logits_fake
        
        # 只有当能量差距不够大时才产生 Loss
        is_real = (self.label == 0)
        is_fake = (self.label == 1)
        
        # Hinge Loss:
        # Real: max(0, 0.5 - (L_real - L_fake)) -> 强迫 L_real 比 L_fake 大 0.5
        loss_real = torch.clamp(self.margin - diff[is_real], min=0).mean()
        
        # Fake: max(0, 0.5 + (L_real - L_fake)) -> 强迫 L_fake 比 L_real 大 0.5
        loss_fake = torch.clamp(self.margin + diff[is_fake], min=0).mean()
        
        ebm_loss = loss_real + loss_fake
        
        # 组合 Loss
        return ce_loss + self.lambda_ebm * ebm_loss

    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss() # 使用新的 Loss 计算逻辑
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
