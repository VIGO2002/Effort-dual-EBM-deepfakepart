import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectors import DETECTOR
from transformers import CLIPModel
from metrics.base_metrics_class import calculate_metrics_for_train

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        
        # 1. Backbone: 保持不变，这是 Effort 的核心 (CLIP + SVD)
        self.backbone = self.build_backbone(config)
        
        # 2. Head: 【修改点】变回普通的线性分类层
        # CLIP ViT-Large 的输出维度是 1024，分类数为 2 (Real/Fake)
        self.head = nn.Linear(1024, 2) 

        # 3. Loss权重: 【修改点】只需要正交损失，不需要 energy/smooth loss
        self.lambda_ortho = 0.5

    def build_backbone(self, config):
        # 路径请根据你的实际情况确认
        model_path = "/root/autodl-tmp/pretrained_models/clip-vit-large-patch14"
        try:
            clip_model = CLIPModel.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load CLIP model. Error: {e}")
            raise e
        
        # 应用 SVD (Effort 核心，必须保留)
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024 - 1)
        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> dict:
        # 【修改点】标准的线性分类
        logits = self.head(features)
        return {'logits': logits}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        logits = pred_dict['logits']
        
        # 1. 标准交叉熵损失
        loss_cls = F.cross_entropy(logits, label)
        
        # 2. SVD 正交损失 (Effort 原文要求)
        loss_ortho = torch.tensor(0.0).to(logits.device)
        if self.training:
            reg_term = 0.0
            num_reg = 0
            for module in self.backbone.modules():
                if isinstance(module, SVDResidualLinear):
                    reg_term += module.compute_orthogonal_loss()
                    reg_term += module.compute_keepsv_loss()
                    num_reg += 1
            if num_reg > 0:
                loss_ortho = self.lambda_ortho * reg_term / num_reg
        
        total_loss = loss_cls + loss_ortho
        
        return {'overall': total_loss, 'cls': loss_cls, 'ortho': loss_ortho}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        logits = pred_dict['logits']
        # 计算标准指标
        auc, eer, acc, ap = calculate_metrics_for_train(label, logits)
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        out = self.classifier(features)
        logits = out['logits']
        # 假图概率 (Label 1)
        prob = F.softmax(logits, dim=1)[:, 1]
        
        return {'logits': logits, 'prob': prob, 'feat': features}

# ==========================================================
# 下面的 SVD 代码块保持完全不变
# ==========================================================
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None
        self.S_r = None
        self.U_r = None
        self.V_r = None
        self.weight_original_fnorm = None

    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)

    def compute_orthogonal_loss(self):
        if self.S_residual is not None:
            UUT = torch.cat((self.U_r, self.U_residual), dim=1) @ torch.cat((self.U_r, self.U_residual), dim=1).t()
            VVT = torch.cat((self.V_r, self.V_residual), dim=0) @ torch.cat((self.V_r, self.V_residual), dim=0).t()
            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0
        return loss

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            loss = torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
        else:
            loss = 0.0
        return loss

def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            apply_svd_residual_to_self_attn(module, r)
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
        r = min(r, len(S))
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')
        new_module.weight_main.data.copy_(weight_main)
        U_residual = U[:, r:]
        S_residual = S[r:]
        Vh_residual = Vh[r:, :]
        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None
        return new_module
    else:
        return module