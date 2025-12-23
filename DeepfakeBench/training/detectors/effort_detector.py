import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import loralib as lora
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)


# ==========================================
# 核心模块: 双能量头 (Dual Energy Head)
# ==========================================
class DualEnergyHead(nn.Module):
    """
    双能量网络：分别建模真实和伪造图像的能量流形。
    理论支撑：E_real(x) 衡量样本与真实流形的偏离度，E_fake(x) 衡量样本与伪造流形的偏离度。
    """
    def __init__(self, in_features, hidden_dim=512, init_noise_std=0.01):
        super().__init__()
        
        # 共享特征提取层 (Feature Extractor)
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),       # 归一化有助于EBM稳定
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)                 # 防止对特定能量点过拟合
        )
        
        # 真实图像能量分支 (Energy Branch for Real)
        self.energy_real = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim//2, 1)  # 输出标量 E_real(x)
        )
        
        # 伪造图像能量分支 (Energy Branch for Fake)
        self.energy_fake = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim//2, 1)  # 输出标量 E_fake(x)
        )
        
        # 【亮点】可学习的噪声强度参数 (Adaptive Manifold Smoothing)
        # 初始值设为 0.01，让模型自己学习需要多大的扰动
        # 2. 【修改】使用传入的参数初始化，而不是写死 0.01
        self.noise_std = nn.Parameter(torch.tensor(init_noise_std))
        
    def forward(self, x, training=True):
        # 特征编码
        h = self.shared_encoder(x)
        
        # 训练时注入噪声 (Langevin Dynamics Approximation)
        # 限制噪声范围在 [0.001, 0.1] 之间，防止训练初期噪声过大导致崩塌
        if training:
            # 原代码: max=0.1
            # 建议修改: max=0.05 (在 forward 函数里)
            std = torch.clamp(self.noise_std, min=0.001, max=0.05)
            noise = torch.randn_like(h) * std
            h_noisy = h + noise
            
            # 计算噪声版本的能量（用于一致性正则 Loss）
            e_real_noisy = self.energy_real(h_noisy)
            e_fake_noisy = self.energy_fake(h_noisy)
        else:
            h_noisy = h
            e_real_noisy = None
            e_fake_noisy = None
        
        # 计算原始特征的能量值
        e_real = self.energy_real(h)
        e_fake = self.energy_fake(h)
        
        # 【关键修正】分类 Logits
        # DeepfakeBench: Label 0 = Real, Label 1 = Fake
        # P(y=0) \propto exp(-E_real) -> Logit[0] = -E_real
        # P(y=1) \propto exp(-E_fake) -> Logit[1] = -E_fake
        logits = torch.cat([-e_real, -e_fake], dim=1)  # shape: (B, 2)
        
        if training:
            return logits, e_real, e_fake, e_real_noisy, e_fake_noisy
        else:
            return logits, e_real, e_fake


# ==========================================
# 辅助模块: 能量监控器
# ==========================================
class EnergyMonitor:
    """用于在训练过程中监控能量分布的统计指标"""
    @staticmethod
    def compute_metrics(pred_dict, labels):
        metrics = {}
        e_real = pred_dict.get('e_real', None)
        e_fake = pred_dict.get('e_fake', None)
        
        if e_real is not None and e_fake is not None:
            # 1. 能量均值 (Energy Means)
            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            
            if real_mask.sum() > 0:
                metrics['avg_E_real_on_Real'] = e_real[real_mask].mean().item()
                metrics['avg_E_fake_on_Real'] = e_fake[real_mask].mean().item() # 应该很高
            
            if fake_mask.sum() > 0:
                metrics['avg_E_fake_on_Fake'] = e_fake[fake_mask].mean().item()
                metrics['avg_E_real_on_Fake'] = e_real[fake_mask].mean().item() # 应该很高
            
            # 2. 能量间隔 (Energy Margin) - 越大越好
            # 对于 Fake: E_real - E_fake 应该 > 0
            if fake_mask.sum() > 0:
                margin_fake = (e_real[fake_mask] - e_fake[fake_mask]).mean().item()
                metrics['margin_fake'] = margin_fake
                
            # 对于 Real: E_fake - E_real 应该 > 0
            if real_mask.sum() > 0:
                margin_real = (e_fake[real_mask] - e_real[real_mask]).mean().item()
                metrics['margin_real'] = margin_real
        
        return metrics


@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)

        # 1. 【修改】从 config 读取 noise_std，如果 YAML 没写，默认用 0.01
        # 这就和你 YAML 里的 "noise_std: 0.02" 对应上了
        initial_noise = config.get('noise_std', 0.01)

        # 2. 【修改】把这个参数传给 DualEnergyHead
        self.head = DualEnergyHead(
            in_features=1024, 
            hidden_dim=512, 
            init_noise_std=initial_noise  # <--- 传进去
        )

        # 初始化监控器
        self.energy_monitor = EnergyMonitor()
        
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        # === 核心超参数 ===
        # Margin 设为 5.0 即可，因为这里是相对比较 (E_real vs E_fake)
        # 相对比较比绝对值比较容易训练，不需要 10.0 那么大
        self.energy_margin = 5.0  
        
        # Loss 权重 (配合 lr=1e-4)
        self.lambda_energy = 0.5  # 能量间隔损失权重
        self.lambda_smooth = 0.1  # 平滑损失权重
        self.lambda_ortho = 0.5   # SVD 正交损失权重

    def build_backbone(self, config):
        # 请确保路径正确
        model_path = "/root/autodl-tmp/pretrained_models/clip-vit-large-patch14"
        try:
            clip_model = CLIPModel.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load CLIP model. Error: {e}")
            raise e

        # 应用 SVD (Effort 核心)
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024 - 1)
        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor, training: bool = False) -> dict:
        if training:
            logits, e_real, e_fake, e_real_noisy, e_fake_noisy = self.head(features, training=True)
            return {
                'logits': logits,
                'e_real': e_real,
                'e_fake': e_fake,
                'e_real_noisy': e_real_noisy,
                'e_fake_noisy': e_fake_noisy
            }
        else:
            logits, e_real, e_fake = self.head(features, training=False)
            return {
                'logits': logits,
                'e_real': e_real,
                'e_fake': e_fake
            }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        
        logits = pred_dict['logits']
        e_real = pred_dict['e_real'].squeeze()
        e_fake = pred_dict['e_fake'].squeeze()
        
        # --- 1. Cross Entropy Loss (主导分类任务) ---
        loss_cls = F.cross_entropy(logits, label)
        
        # --- 2. Energy Margin Loss (塑造流形) ---
        # 目的：拉大正确类与错误类能量的差距
        margin = self.energy_margin
        
        # Fake 样本 (label=1): 希望 E_fake < E_real - margin
        # 即 (E_fake - E_real + margin) 应该 < 0
        fake_mask = (label == 1)
        loss_fake_energy = 0.0
        if fake_mask.sum() > 0:
            loss_fake_energy = F.relu(e_fake[fake_mask] - e_real[fake_mask] + margin).mean()
        
        # Real 样本 (label=0): 希望 E_real < E_fake - margin
        real_mask = (label == 0)
        loss_real_energy = 0.0
        if real_mask.sum() > 0:
            loss_real_energy = F.relu(e_real[real_mask] - e_fake[real_mask] + margin).mean()
            
        loss_energy = loss_fake_energy + loss_real_energy
        
        # --- 3. Stability/Smoothness Loss (基于噪声) ---
        loss_smooth = 0.0
        if self.training and 'e_real_noisy' in pred_dict:
            e_real_noisy = pred_dict['e_real_noisy'].squeeze()
            e_fake_noisy = pred_dict['e_fake_noisy'].squeeze()
            # 强迫能量函数在局部邻域内保持平滑
            loss_smooth = F.mse_loss(e_real, e_real_noisy) + F.mse_loss(e_fake, e_fake_noisy)
        
        # --- 4. Orthogonal Loss (Effort 原文) ---
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
        
        # --- 总损失 ---
        total_loss = (loss_cls + 
                      self.lambda_energy * loss_energy + 
                      self.lambda_smooth * loss_smooth + 
                      loss_ortho)
        
        # 数值保护
        if torch.isnan(total_loss):
            total_loss = loss_cls # 降级策略
            
        loss_dict = {
            'overall': total_loss,
            'cls': loss_cls,
            'energy': loss_energy,
            'smooth': loss_smooth,
            'ortho': loss_ortho
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        # 修改点：直接使用 logits (Tensor)，不要用 prob，也不要转 numpy
        # calculate_metrics_for_train 内部需要 Tensor 来做 torch.max 和 .size(1) 操作
        logits = pred_dict['logits']
        
        auc, eer, acc, ap = calculate_metrics_for_train(
            label,   # 传入 Tensor
            logits   # 传入 Tensor (B, 2)
        )
        
        # 额外记录能量统计信息
        energy_metrics = self.energy_monitor.compute_metrics(pred_dict, label)
        
        metric_batch_dict = {
            'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap,
            **energy_metrics  # 合并能量统计信息
        }
        
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        out = self.classifier(features, training=not inference)
        
        logits = out['logits']
        # 计算概率用于指标评估
        prob = F.softmax(logits, dim=1)[:, 1]
        
        pred_dict = {
            'logits': logits,
            'prob': prob,
            'feat': features,
            'e_real': out['e_real'],
            'e_fake': out['e_fake']
        }
        
        if not inference:
            pred_dict['e_real_noisy'] = out.get('e_real_noisy')
            pred_dict['e_fake_noisy'] = out.get('e_fake_noisy')
            
        return pred_dict


# ==========================================
# SVD 组件 (保持不变)
# ==========================================
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

    def compute_fn_loss(self):
        if (self.S_residual is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            loss = weight_current_fnorm ** 2
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