"""
几何缺陷分支模型
提取几何特征进行空间一致性验证
"""

import math
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from typing import Tuple


class GeometricBranch:
    """
    几何缺陷分支：验证空间一致性与局部差异
    
    特征：
    - 极坐标分bin特征 (36维)：径向3 × 角度12
    - 网格纹理特征 (64维)：多尺度4×4, 8×8
    - 环形差异特征 (28维)：环形区域对比
    
    总计：128维几何特征
    """
    
    def __init__(self, canvas_size=256):
        self.canvas_size = canvas_size
        self.feature_dim = 128
    
    def extract_features(self, mask):
        """
        提取几何特征
        
        Args:
            mask: 二值化的印章掩膜 [H, W]
            
        Returns:
            features: 128维几何特征向量
        """
        # 归一化印章
        normalized = self._normalize_seal(mask)
        
        # 提取各类特征
        features_list = []
        
        # 1. 极坐标分bin特征 (36维)
        polar_features = self._extract_polar_features(normalized, radial_bins=3, angular_bins=12)
        features_list.append(polar_features)
        
        # 2. 网格纹理特征 (64维)
        grid_features = self._extract_grid_features(normalized, grid_sizes=[4, 8])
        features_list.append(grid_features)
        
        # 3. 环形差异特征 (28维)
        ring_features = self._extract_ring_features(normalized, angular_bins=4)
        features_list.append(ring_features)
        
        # 合并特征
        features = np.concatenate(features_list).astype(np.float32)
        
        # 验证特征维度
        expected_dim = 36 + 64 + 28  # 128
        if len(features) != expected_dim:
            raise ValueError(f"特征维度错误: 期望{expected_dim}, 实际{len(features)}")
        
        return features
    
    def _normalize_seal(self, mask):
        """
        归一化印章到标准画布
        
        Args:
            mask: 原始印章掩膜
            
        Returns:
            normalized_mask: 归一化后的掩膜 [canvas_size, canvas_size]
        """
        ys, xs = np.where(mask > 0)
        
        # 更严格的像素数检查
        if xs.size < 500:
            # 像素太少（小于500），返回空掩膜
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        # 计算边界
        boundary = mask.astype(bool) & (~ndi.binary_erosion(mask.astype(bool), iterations=1))
        bys, bxs = np.where(boundary)
        
        if bxs.size == 0:
            # 没有边界，使用所有点
            bxs, bys = xs, ys
        
        boundary_coords = np.column_stack([bxs.astype(np.float32), bys.astype(np.float32)])
        
        # 计算中心和主轴
        coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        center = coords.mean(axis=0)
        
        # 计算协方差矩阵
        centered = coords - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        # 计算旋转角度
        angle = math.atan2(float(eigvecs[1, 0]), float(eigvecs[0, 0]))
        
        # 计算轴长
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        axis_a = max((x1 - x0 + 1.0) / 2.0, 1.0)
        axis_b = max((y1 - y0 + 1.0) / 2.0, 1.0)
        
        # 归一化坐标
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dx = coords[:, 0] - center[0]
        dy = coords[:, 1] - center[1]
        xr = cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy
        corrected = np.column_stack([xr / max(axis_a, 1e-3), yr / max(axis_b, 1e-3)])
        
        # 映射到画布
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        scale = (self.canvas_size - 1) / 2.0
        x_img = np.clip(np.round(corrected[:, 0] * scale + scale).astype(int), 0, self.canvas_size - 1)
        y_img = np.clip(np.round(corrected[:, 1] * scale + scale).astype(int), 0, self.canvas_size - 1)
        canvas[y_img, x_img] = 1
        
        # 闭运算和填充
        canvas = ndi.binary_closing(canvas, iterations=2)
        canvas = ndi.binary_fill_holes(canvas)
        
        return canvas.astype(np.uint8)
    
    def _extract_polar_features(self, mask, radial_bins=3, angular_bins=12):
        """提取极坐标分bin特征"""
        side = mask.shape[0]
        
        # 创建极坐标网格
        coords = (np.arange(side, dtype=np.float32) + 0.5 - side / 2.0) / (side / 2.0)
        x_grid, y_grid = np.meshgrid(coords, coords)
        radius = np.sqrt(x_grid**2 + y_grid**2)
        theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
        
        # 分bin统计
        radial_edges = np.linspace(0.0, 1.0, radial_bins + 1)
        angular_edges = np.linspace(0.0, 2.0 * np.pi, angular_bins + 1)
        
        features = []
        for ridx in range(radial_bins):
            r0, r1 = radial_edges[ridx], radial_edges[ridx + 1]
            radial_region = (radius >= r0) & (radius < r1)
            
            for aidx in range(angular_bins):
                a0, a1 = angular_edges[aidx], angular_edges[aidx + 1]
                region = radial_region & (theta >= a0) & (theta < a1)
                denom = int(region.sum())
                value = 0.0 if denom == 0 else float(mask[region].mean())
                features.append(value)
        
        return np.asarray(features, dtype=np.float32)
    
    def _extract_grid_features(self, mask, grid_sizes=[4, 8]):
        """
        提取网格纹理特征
        
        返回64维特征：
        - 4×4网格：16个单元密度 = 16维
        - 8×8网格：前48个单元密度 = 48维
        总计：64维
        """
        values = []
        
        for m in grid_sizes:
            side = mask.shape[0]
            cell_edges = np.linspace(0, side, m + 1, dtype=int)
            
            # 确定需要提取的单元数
            if m == 4:
                max_cells = 16  # 4×4 = 16
            elif m == 8:
                max_cells = 48  # 取前48个
            else:
                max_cells = m * m
            
            cell_count = 0
            # 提取每个网格单元的密度
            for row in range(m):
                for col in range(m):
                    if cell_count >= max_cells:
                        break
                    
                    y0, y1 = cell_edges[row], cell_edges[row + 1]
                    x0, x1 = cell_edges[col], cell_edges[col + 1]
                    cell_density = float(mask[y0:y1, x0:x1].mean())
                    values.append(cell_density)
                    cell_count += 1
                
                if cell_count >= max_cells:
                    break
        
        result = np.asarray(values, dtype=np.float32)
        
        # 验证维度
        if len(result) != 64:
            raise ValueError(f"网格特征维度错误: 期望64, 实际{len(result)}")
        
        return result
    
    def _extract_ring_features(self, mask, angular_bins=4):
        """
        提取环形差异特征
        
        返回28维特征：
        - 3个环 × 4个角度 = 12维（各环的角度分布）
        - 3个环的统计特征 = 6维（均值、标准差）
        - 环间差异 = 10维（3个环两两比较的统计）
        总计：28维
        """
        side = mask.shape[0]
        coords = (np.arange(side, dtype=np.float32) + 0.5 - side / 2.0) / (side / 2.0)
        x_grid, y_grid = np.meshgrid(coords, coords)
        radius = np.sqrt(x_grid**2 + y_grid**2)
        theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
        
        # 定义3个环
        ring_edges = [0.0, 0.33, 0.67, 1.0]
        angular_edges = np.linspace(0.0, 2.0 * np.pi, angular_bins + 1)
        
        features = []
        ring_profiles = []
        
        # 提取每个环的角度分布
        for ridx in range(3):
            r0, r1 = ring_edges[ridx], ring_edges[ridx + 1]
            radial_region = (radius >= r0) & (radius < r1)
            
            ring_angular = []
            for aidx in range(angular_bins):
                a0, a1 = angular_edges[aidx], angular_edges[aidx + 1]
                region = radial_region & (theta >= a0) & (theta < a1)
                denom = int(region.sum())
                value = 0.0 if denom == 0 else float(mask[region].mean())
                ring_angular.append(value)
            
            features.extend(ring_angular)  # 12维
            ring_profiles.append(ring_angular)
        
        # 计算每个环的统计特征
        for profile in ring_profiles:
            features.append(float(np.mean(profile)))  # 均值
            features.append(float(np.std(profile)))   # 标准差
        # 6维
        
        # 计算环间差异（两两比较）
        for i in range(3):
            for j in range(i + 1, 3):
                diff = np.array(ring_profiles[i]) - np.array(ring_profiles[j])
                features.append(float(np.mean(np.abs(diff))))  # 平均绝对差异
                features.append(float(np.std(diff)))           # 差异的标准差
        # 6维（3对环）
        
        # 添加额外的全局差异特征
        all_profiles = np.array(ring_profiles)
        features.append(float(np.mean(all_profiles)))  # 全局均值
        features.append(float(np.std(all_profiles)))   # 全局标准差
        features.append(float(np.max(all_profiles) - np.min(all_profiles)))  # 范围
        features.append(float(np.mean(np.abs(np.diff(all_profiles, axis=0)))))  # 环间梯度
        # 4维
        
        return np.asarray(features, dtype=np.float32)  # 总计28维


if __name__ == "__main__":
    # 测试几何分支
    branch = GeometricBranch()
    
    # 创建测试掩膜（圆形）
    size = 256
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 3
    mask = ((x - center)**2 + (y - center)**2 <= radius**2).astype(np.uint8)
    
    # 提取特征
    features = branch.extract_features(mask)
    print(f"特征维度: {features.shape}")
    print(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")
    print(f"非零特征数: {np.sum(features != 0)}")
