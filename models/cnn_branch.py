"""
CNN分支模型
使用ResNet-50提取全局特征和局部细节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from torchvision import models


def _build_resnet50(pretrained=True):
    """兼容新旧 torchvision 的 ResNet-50 构造。"""
    signature = inspect.signature(models.resnet50)
    if 'weights' in signature.parameters:
        weights = None
        if pretrained:
            weights_enum = getattr(models, 'ResNet50_Weights', None)
            if weights_enum is not None:
                weights = getattr(weights_enum, 'DEFAULT', None) or getattr(weights_enum, 'IMAGENET1K_V1', None)
        return models.resnet50(weights=weights)

    # 老版本 torchvision 仅支持 pretrained 参数
    return models.resnet50(pretrained=pretrained)


class CNNBranch(nn.Module):
    """
    CNN主分支：提取全局结构与局部细节特征
    
    架构：
    - 骨干网络：ResNet-50 (ImageNet预训练)
    - 全局特征：全局平均池化
    - 输出：2048维特征 + 2分类
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(CNNBranch, self).__init__()
        
        # 加载预训练的ResNet-50
        self.backbone = _build_resnet50(pretrained=pretrained)
        
        # 获取特征维度
        self.feature_dim = self.backbone.fc.in_features  # 2048
        
        # 移除原始的全连接层
        self.backbone.fc = nn.Identity()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, return_features=False):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, 224, 224]
            return_features: 是否返回特征向量
            
        Returns:
            如果return_features=True: (logits, features)
            否则: logits
        """
        # 提取特征
        features = self.backbone(x)  # [B, 2048]
        
        # 分类
        logits = self.classifier(features)  # [B, num_classes]
        
        if return_features:
            return logits, features
        return logits
    
    def extract_features(self, x):
        """仅提取特征向量"""
        with torch.no_grad():
            features = self.backbone(x)
        return features


class CNNBranchWithAttention(nn.Module):
    """
    带注意力机制的CNN分支（可选增强版本）
    
    增加了通道注意力和空间注意力
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(CNNBranchWithAttention, self).__init__()
        
        # 加载预训练的ResNet-50
        resnet = _build_resnet50(pretrained=pretrained)
        
        # 提取特征提取部分（去掉avgpool和fc）
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 通道注意力
        self.channel_attention = ChannelAttention(2048)
        
        # 空间注意力
        self.spatial_attention = SpatialAttention()
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, return_features=False):
        # 基础特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, 7, 7]
        
        # 应用注意力
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        
        # 全局池化
        features = self.avgpool(x)
        features = torch.flatten(features, 1)  # [B, 2048]
        
        # 分类
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # 融合
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return out


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 通道维度的平均和最大
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


def create_cnn_branch(num_classes=2, pretrained=True, use_attention=False):
    """
    创建CNN分支模型的工厂函数
    
    Args:
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        use_attention: 是否使用注意力机制
        
    Returns:
        model: CNN分支模型
    """
    if use_attention:
        return CNNBranchWithAttention(num_classes, pretrained)
    else:
        return CNNBranch(num_classes, pretrained)


if __name__ == "__main__":
    # 测试模型
    model = create_cnn_branch(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    
    # 测试前向传播
    logits = model(x)
    print(f"Logits shape: {logits.shape}")
    
    # 测试特征提取
    logits, features = model(x, return_features=True)
    print(f"Features shape: {features.shape}")
    
    # 测试注意力版本
    model_att = create_cnn_branch(num_classes=2, pretrained=False, use_attention=True)
    logits_att = model_att(x)
    print(f"Attention model logits shape: {logits_att.shape}")
