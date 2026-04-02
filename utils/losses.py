"""
损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出logits [B, C]
            targets: 目标标签 [B]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CenterLoss(nn.Module):
    """
    Center Loss
    用于增强类内紧凑性
    """
    
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # 类中心（不在初始化时指定device，由外部.to(device)处理）
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features, labels):
        """
        Args:
            features: 特征向量 [B, feat_dim]
            labels: 标签 [B]
        """
        batch_size = features.size(0)
        
        # 计算特征到中心的距离
        centers_batch = self.centers.index_select(0, labels.long())
        loss = (features - centers_batch).pow(2).sum() / batch_size
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss
    用于度量学习
    """
    
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: 锚点特征 [B, D]
            positive: 正样本特征 [B, D]
            negative: 负样本特征 [B, D]
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    组合损失
    结合分类损失和度量损失
    """
    
    def __init__(self, 
                 num_classes=2,
                 feat_dim=2048,
                 use_focal=True,
                 use_center=False,
                 alpha_focal=0.75,
                 gamma_focal=2.0,
                 lambda_center=0.1,
                 device='cuda'):
        super(CombinedLoss, self).__init__()
        
        # 分类损失
        if use_focal:
            self.cls_loss = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)
        else:
            self.cls_loss = nn.CrossEntropyLoss()
        
        # Center Loss
        self.use_center = use_center
        if use_center:
            self.center_loss = CenterLoss(num_classes, feat_dim, device)
            self.lambda_center = lambda_center
    
    def forward(self, logits, features, labels):
        """
        Args:
            logits: 分类logits [B, num_classes]
            features: 特征向量 [B, feat_dim]
            labels: 标签 [B]
        """
        # 分类损失
        loss_cls = self.cls_loss(logits, labels)
        
        # Center Loss
        if self.use_center:
            loss_center = self.center_loss(features, labels)
            total_loss = loss_cls + self.lambda_center * loss_center
            return total_loss, loss_cls, loss_center
        else:
            return loss_cls, loss_cls, torch.tensor(0.0)


if __name__ == "__main__":
    # 测试Focal Loss
    focal_loss = FocalLoss()
    inputs = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1])
    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # 测试Center Loss
    center_loss = CenterLoss(num_classes=2, feat_dim=128, device='cpu')
    features = torch.randn(4, 128)
    labels = torch.tensor([0, 1, 0, 1])
    loss = center_loss(features, labels)
    print(f"Center Loss: {loss.item():.4f}")
    
    # 测试Combined Loss
    combined_loss = CombinedLoss(num_classes=2, feat_dim=128, use_center=True, device='cpu')
    logits = torch.randn(4, 2)
    features = torch.randn(4, 128)
    labels = torch.tensor([0, 1, 0, 1])
    total_loss, cls_loss, center_loss = combined_loss(logits, features, labels)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Cls Loss: {cls_loss.item():.4f}")
    print(f"Center Loss: {center_loss.item():.4f}")
