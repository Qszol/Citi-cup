"""
多分支融合模型
整合CNN、物理、几何三个分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

from .cnn_branch import CNNBranch
from .physical_branch import PhysicalBranch
from .geometric_branch import GeometricBranch


class SealAuthenticationModel(nn.Module):
    """
    印章真假鉴别融合模型
    
    架构：
    1. CNN分支：2048维深度特征
    2. 物理分支：33维显式特征
    3. 几何分支：128维几何特征
    4. 融合层：特征拼接 + 全连接
    """
    
    def __init__(self, 
                 num_classes=2,
                 cnn_pretrained=True,
                 fusion_hidden=512,
                 dropout_rate=0.3,
                 cnn_bottleneck_dim=512,
                 cnn_feature_weight=1.0,
                 physical_feature_weight=0.35,
                 geometric_feature_weight=0.35):
        super(SealAuthenticationModel, self).__init__()
        
        # CNN分支
        self.cnn_branch = CNNBranch(num_classes=num_classes, pretrained=cnn_pretrained)
        
        # 特征维度
        self.cnn_dim = 2048 # ResNet50 的默认输出
        self.physical_dim = 33
        self.geometric_dim = 128
        
        # CNN主导：保留更高维视觉表征，并降低显式特征分支的相对影响
        self.cnn_bottleneck = nn.Sequential(
            nn.Linear(self.cnn_dim, cnn_bottleneck_dim),
            nn.BatchNorm1d(cnn_bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.cnn_feature_weight = cnn_feature_weight
        self.physical_feature_weight = physical_feature_weight
        self.geometric_feature_weight = geometric_feature_weight
        
        self.total_dim = cnn_bottleneck_dim + self.physical_dim + self.geometric_dim
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.total_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(fusion_hidden // 2, num_classes)
        )
    
    def forward(self, 
                cnn_input, 
                physical_features=None, 
                geometric_features=None,
                return_branch_outputs=False):
        """
        前向传播
        
        Args:
            cnn_input: CNN输入图像 [B, 3, 224, 224]
            physical_features: 物理特征 [B, 33] (可选)
            geometric_features: 几何特征 [B, 128] (可选)
            return_branch_outputs: 是否返回各分支输出
            
        Returns:
            logits: 融合后的分类logits [B, num_classes]
            如果return_branch_outputs=True，还返回各分支的输出
        """
        batch_size = cnn_input.size(0)
        
        # CNN分支
        cnn_logits, cnn_features = self.cnn_branch(cnn_input, return_features=True)
        
        # 如果没有提供物理和几何特征，使用零向量
        if physical_features is None:
            physical_features = torch.zeros(batch_size, self.physical_dim, 
                                           device=cnn_input.device)
        
        if geometric_features is None:
            geometric_features = torch.zeros(batch_size, self.geometric_dim,
                                            device=cnn_input.device)
        
        # 特征拼接（CNN主导）
        cnn_bottleneck_features = self.cnn_bottleneck(cnn_features)
        fused_features = torch.cat([
            self.cnn_feature_weight * cnn_bottleneck_features,
            self.physical_feature_weight * physical_features,
            self.geometric_feature_weight * geometric_features
        ], dim=1)
        
        # 融合分类
        fusion_logits = self.fusion(fused_features)
        
        if return_branch_outputs:
            return {
                'fusion_logits': fusion_logits,
                'cnn_logits': cnn_logits,
                'cnn_features': cnn_features,
                'physical_features': physical_features,
                'geometric_features': geometric_features,
                'fused_features': fused_features
            }
        
        return fusion_logits


class AdaptiveWeightFusion(nn.Module):
    """
    自适应权重融合模型
    根据各分支的置信度动态调整权重
    """
    
    def __init__(self, num_classes=2):
        super(AdaptiveWeightFusion, self).__init__()
        
        # 权重网络：根据各分支特征学习权重
        self.weight_net = nn.Sequential(
            nn.Linear(2048 + 33 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),  # 3个分支的权重
            nn.Softmax(dim=1)
        )
    
    def forward(self, cnn_logits, physical_score, geometric_score, features):
        """
        自适应权重融合
        
        Args:
            cnn_logits: CNN分支logits [B, 2]
            physical_score: 物理分支得分 [B, 1]
            geometric_score: 几何分支得分 [B, 1]
            features: 拼接的特征 [B, 2209]
            
        Returns:
            fusion_logits: 融合后的logits [B, 2]
        """
        # 学习权重
        weights = self.weight_net(features)  # [B, 3]
        
        # 将得分转换为logits形式
        physical_logits = torch.stack([1 - physical_score, physical_score], dim=1).squeeze(-1)
        geometric_logits = torch.stack([1 - geometric_score, geometric_score], dim=1).squeeze(-1)
        
        # 加权融合
        cnn_prob = F.softmax(cnn_logits, dim=1)
        physical_prob = F.softmax(physical_logits, dim=1)
        geometric_prob = F.softmax(geometric_logits, dim=1)
        
        # 加权平均
        fusion_prob = (weights[:, 0:1] * cnn_prob + 
                      weights[:, 1:2] * physical_prob + 
                      weights[:, 2:3] * geometric_prob)
        
        # 转回logits
        fusion_logits = torch.log(fusion_prob + 1e-10)
        
        return fusion_logits, weights


def create_fusion_model(model_type='basic', **kwargs):
    """
    创建融合模型的工厂函数
    
    Args:
        model_type: 模型类型
            - 'basic': 基础融合模型
            - 'adaptive': 自适应权重融合
        **kwargs: 其他参数
        
    Returns:
        model: 融合模型
    """
    if model_type == 'basic':
        return SealAuthenticationModel(**kwargs)
    elif model_type == 'adaptive':
        return AdaptiveWeightFusion(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试融合模型
    model = create_fusion_model(model_type='basic', num_classes=2, cnn_pretrained=False)
    
    # 创建测试数据
    batch_size = 4
    cnn_input = torch.randn(batch_size, 3, 224, 224)
    physical_features = torch.randn(batch_size, 33)
    geometric_features = torch.randn(batch_size, 128)
    
    # 测试前向传播
    logits = model(cnn_input, physical_features, geometric_features)
    print(f"Fusion logits shape: {logits.shape}")
    
    # 测试返回各分支输出
    outputs = model(cnn_input, physical_features, geometric_features, 
                   return_branch_outputs=True)
    print(f"CNN logits shape: {outputs['cnn_logits'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    # 测试自适应权重融合
    adaptive_model = create_fusion_model(model_type='adaptive', num_classes=2)
    cnn_logits = torch.randn(batch_size, 2)
    physical_score = torch.rand(batch_size, 1)
    geometric_score = torch.rand(batch_size, 1)
    features = torch.randn(batch_size, 2209)
    
    fusion_logits, weights = adaptive_model(cnn_logits, physical_score, 
                                           geometric_score, features)
    print(f"Adaptive fusion logits shape: {fusion_logits.shape}")
    print(f"Learned weights shape: {weights.shape}")
    print(f"Example weights: {weights[0]}")
