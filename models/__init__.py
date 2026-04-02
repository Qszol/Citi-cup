"""
模型模块
"""

from .cnn_branch import CNNBranch, CNNBranchWithAttention, create_cnn_branch
from .physical_branch import PhysicalBranch
from .geometric_branch import GeometricBranch
from .fusion_model import SealAuthenticationModel, AdaptiveWeightFusion, create_fusion_model

__all__ = [
    'CNNBranch',
    'CNNBranchWithAttention',
    'create_cnn_branch',
    'PhysicalBranch',
    'GeometricBranch',
    'SealAuthenticationModel',
    'AdaptiveWeightFusion',
    'create_fusion_model',
]
