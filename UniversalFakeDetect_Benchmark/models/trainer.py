import functools
import torch
import torch.nn as nn
from .base_model import BaseModel, init_weights
import sys
from models import get_model
from transformers import get_cosine_schedule_with_warmup # 建议引入 transformers 的 scheduler，如果没有安装可以换成 torch 原生的

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt.arch, opt)
        self.lr = opt.lr
        
        # [修改 1] EBM_Head 初始化逻辑优化
        # 遍历 fc 模块下的所有子模块，如果是 Linear 则初始化
        if hasattr(self.model, 'fc'):
            for m in self.model.fc.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0.0, opt.init_gain)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)

        # [修改 2] 参数冻结逻辑适配 Dual-EBM
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                # 只要参数名包含 'fc.' (比如 fc.energy_real.0.weight)，就认为是分类头，需要训练
                if 'fc.' in name: 
                    params.append(p) 
                    p.requires_grad = True
                # SVD 残差参数也需要训练 (如果有)
                elif any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                    params.append(p)
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            print(f">>> Backbone fixed. Training {len(params)} tensors (Head + SVD Residuals).")
        else:
            print("Your backbone is not fixed. Training all parameters.")
            params = self.model.parameters()

        # 优化器设置
        if opt.optim == 'adam':
            # 这里的 beta2 设为 0.999 是标准的，DeepfakeBench 里可能用了 AdamW
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        # [修改 3] Loss 函数：使用 CrossEntropy
        # 因为 Dual-EBM 输出的是 [-E_real, -E_fake]，这正好对应 Logits
        # CrossEntropy 会自动做 Softmax，符合 P(y) = exp(-E_y) / Z 的能量分布公式
        self.loss_fn = nn.CrossEntropyLoss()

        self.model.to(opt.gpu_ids[0])
        
        # [新增] Warmup Scheduler
        # 如果 opt 中定义了 warmup_steps，则初始化调度器
        self.scheduler = None
        if hasattr(opt, 'warmup_steps') and opt.warmup_steps > 0:
            print(f">>> Using Cosine Scheduler with {opt.warmup_steps} warmup steps.")
            # 这里的 num_training_steps 需要估算，或者在 train.py 里传入
            # 这里简单给一个较大的值，或者只用 constant warmup
            try:
                # 尝试计算总步数：epoch * iter_per_epoch
                # 由于 Trainer 不知道 data_loader 长度，这里我们用一个比较通用的策略
                # 或者只在 step 阶段手动实现简单的 warmup
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=opt.warmup_steps, num_training_steps=opt.niter * 1000 # 估算值，或者由外部调用
                )
            except:
                pass # 如果没装 transformers 库，就忽略

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).long() # Label 必须是 long 类型

    def forward(self):
        # [修改 4] 移除 view(-1).unsqueeze(1)
        # Dual-EBM 输出 shape 为 [Batch, 2]，直接保留即可
        self.output = self.model(self.input)
        # 确保 output 是 [B, 2]
        if self.output.dim() == 1:
             # 万一只有一个输出，兼容处理
             pass 

    def get_loss(self):
        return self.loss_fn(self.output, self.label)

    def optimize_parameters(self):
        self.forward()
        
        # 计算 Loss
        self.loss = self.loss_fn(self.output, self.label)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        # 更新 Scheduler
        if self.scheduler:
            self.scheduler.step()