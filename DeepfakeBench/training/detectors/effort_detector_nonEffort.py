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

#EBM_HEAD
class EBM_Head(nn.Module):
    def __init__(self, in_features, hidden_features=512):
        super(EBM_Head, self).__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x):
        return self.energy_net(x)

@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        #self.head = nn.Linear(1024, 2)
        #self.loss_func = nn.CrossEntropyLoss()
        self.head = EBM_Head(1024)  # CLIP ViT-L/14 输出维度是 1024
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # ⚠⚠⚠ Download CLIP model using the below link
        # https://drive.google.com/drive/folders/1fm3Jd8lFMiSP1qgdmsxfqlJZGpr_bXsx?usp=drive_link
        
        # mean: [0.48145466, 0.4578275, 0.40821073]
        # std: [0.26862954, 0.26130258, 0.27577711]
        
        # ViT-L/14 224*224
        # 改为官方 ID，让它自动下载/加载缓存
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # the path of this folder in your disk (download from the above link)

        # Apply SVD to self_attn layers only
        # ViT-L/14 224*224: 1024-1
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024-1)

        #for name, param in clip_model.vision_model.named_parameters():
        #    print('{}: {}'.format(name, param.requires_grad))
        #num_param = sum(p.numel() for p in clip_model.vision_model.parameters() if p.requires_grad)
        #num_total_param = sum(p.numel() for p in clip_model.vision_model.parameters())
        #print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    #def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
    #    label = data_dict['label']
    #    pred = pred_dict['cls']
    #    loss = self.loss_func(pred, label)
    #    
    #    if self.training:
    #        # Regularization term
    #        lambda_reg = 1.0
    #        reg_term = 0.0
    #        num_reg = 0
    #        for module in self.backbone.modules():
    #            if isinstance(module, SVDResidualLinear):
    #                reg_term += module.compute_orthogonal_loss()
    #                reg_term += module.compute_keepsv_loss()
    #                num_reg += 1
    #        
    #        loss += lambda_reg * reg_term / num_reg
    #    
    #    loss_dict = {'overall': loss}
    #    return loss_dict

    def compute_weight_loss(self):
        weight_sum_dict = {}
        num_weight_dict = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, SVDResidualLinear):
                weight_curr = module.compute_current_weight()
                if str(weight_curr.size()) not in weight_sum_dict.keys():
                    weight_sum_dict[str(weight_curr.size())] = weight_curr
                    num_weight_dict[str(weight_curr.size())] = 1
                else:
                    weight_sum_dict[str(weight_curr.size())] += weight_curr
                    num_weight_dict[str(weight_curr.size())] += 1
        
        loss2 = 0.0
        for k in weight_sum_dict.keys():
            _, S_sum, _ = torch.linalg.svd(weight_sum_dict[k], full_matrices=False)
            loss2 += -torch.mean(S_sum)
        loss2 /= len(weight_sum_dict.keys())
        return loss2

    # 原代码 (Lines 83-112):
    # def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
    #     ...
    #     loss = self.loss_func(pred, label)
    #     ...

    # 修改为:
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        energy = pred_dict['cls'].squeeze()  # 获取能量值 [Batch]

        # EBM Margin Loss
        # 逻辑：真图(label=0)能量要小于 m_real (-10)
        #       假图(label=1)能量要大于 m_fake (+10)
        m_real = -10.0
        m_fake = 10.0

        # 1. 计算真图损失: max(0, energy - m_real)
        loss_real = torch.clamp(energy[label == 0] - m_real, min=0).mean()
        # 处理 batch 中没有真图的情况
        if torch.isnan(loss_real): loss_real = torch.tensor(0.0).to(energy.device)

        # 2. 计算假图损失: max(0, m_fake - energy)
        loss_fake = torch.clamp(m_fake - energy[label == 1], min=0).mean()
        # 处理 batch 中没有假图的情况
        if torch.isnan(loss_fake): loss_fake = torch.tensor(0.0).to(energy.device)

        loss = loss_real + loss_fake

        # 保留原有的辅助 loss (如果代码里有的话)
        # loss2 = self.compute_weight_loss()
        # overall_loss = loss + loss2

        loss_dict = {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the energy by EBM head
        energy = self.classifier(features)
        # prob 用于计算 AUC。AUC 只需要相对排名。
        # 能量越高越假，所以直接用能量值作为 prob 即可。
        # 为了保持范围在 0-1 (方便 accuracy 计算)，可以用 sigmoid
        prob = torch.sigmoid(energy)

        pred_dict = {'cls': energy, 'prob': prob, 'feat': features}
        return pred_dict


# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)
    
    def compute_orthogonal_loss(self):
        if self.S_residual is not None:
            # According to the properties of orthogonal matrices: A^TA = I
            UUT = torch.cat((self.U_r, self.U_residual), dim=1) @ torch.cat((self.U_r, self.U_residual), dim=1).t()
            VVT = torch.cat((self.V_r, self.V_residual), dim=0) @ torch.cat((self.V_r, self.V_residual), dim=0).t()
            # print(self.U_r.size(), self.U_residual.size())  # torch.Size([1024, 1023]) torch.Size([1024, 1])
            # print(self.V_r.size(), self.V_residual.size())  # torch.Size([1023, 1024]) torch.Size([1, 1024])
            # UUT = self.U_residual @ self.U_residual.t()
            # VVT = self.V_residual @ self.V_residual.t()
            
            # Construct an identity matrix
            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
            
            # Using frobenius norm to compute loss
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0
            
        return loss

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            # Total current weight is the fixed main weight plus the residual
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            
            loss = torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
            # loss = torch.abs(weight_current_fnorm ** 2 + 0.01 * self.weight_main_fnorm ** 2 - 1.01 * self.weight_original_fnorm ** 2)
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

# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]      # Shape: (out_features, r)
        S_r = S[:r]         # Shape: (r,)
        Vh_r = Vh[:r, :]    # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]    # Shape: (out_features, n - r)
        S_residual = S[r:]       # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

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
