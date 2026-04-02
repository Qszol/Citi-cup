"""
印章掩膜提取工具
统一的mask提取方法，供所有模块使用
"""

import numpy as np
import cv2
from scipy import ndimage as ndi


def extract_seal_mask(image_rgb, method='multi_color'):
    """
    提取印章掩膜（支持多种颜色）
    
    Args:
        image_rgb: RGB图像 (H, W, 3)
        method: 提取方法
            - 'multi_color': 多颜色检测（默认，推荐）
            - 'red': 仅红色检测
            - 'hsv': 基于HSV的通用检测
            
    Returns:
        mask: 二值掩膜 (H, W)
    """
    if method == 'red':
        return _extract_red_mask(image_rgb)
    elif method == 'hsv':
        return _extract_hsv_mask(image_rgb)
    else:  # multi_color
        return _extract_multi_color_mask(image_rgb)


def _extract_red_mask(image_rgb):
    """提取红色印章掩膜"""
    image_rgb = image_rgb.astype(np.float32)
    
    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]
    
    # 计算红色得分
    redness = r - 0.5 * (g + b)
    darkness = 255.0 - image_rgb.mean(axis=2)
    contrast = np.maximum(r - g, r - b)
    score = 1.25 * redness + 0.35 * darkness + 0.6 * contrast
    
    # 阈值化
    threshold = np.percentile(score, 90)
    mask = (score > threshold).astype(np.uint8)
    
    return _postprocess_mask(mask)


def _extract_hsv_mask(image_rgb):
    """基于HSV的通用印章检测"""
    hsv = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # 高饱和度 + 低亮度 = 印章
    saturation_score = s.astype(np.float32)
    value_score = 255.0 - v.astype(np.float32)
    score = 0.6 * saturation_score + 0.4 * value_score
    
    # 阈值化
    threshold = np.percentile(score, 90)
    mask = (score > threshold).astype(np.uint8)
    
    return _postprocess_mask(mask)


def _extract_multi_color_mask(image_rgb):
    """
    多颜色印章检测（红色、蓝色、紫色）
    综合多种方法，取最大值
    """
    image_rgb = image_rgb.astype(np.float32)
    
    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]
    
    darkness = 255.0 - image_rgb.mean(axis=2)
    
    # 方法1：红色印章检测
    redness = r - 0.5 * (g + b)
    contrast_r = np.maximum(r - g, r - b)
    red_score = 1.25 * redness + 0.35 * darkness + 0.6 * contrast_r
    
    # 方法2：蓝色印章检测
    blueness = b - 0.5 * (r + g)
    contrast_b = np.maximum(b - r, b - g)
    blue_score = 1.25 * blueness + 0.35 * darkness + 0.6 * contrast_b
    
    # 方法3：紫色印章检测
    purpleness = (r + b) / 2.0 - g
    purple_score = 1.0 * purpleness + 0.5 * darkness
    
    # 方法4：基于HSV的通用检测
    hsv = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    saturation_score = s.astype(np.float32)
    value_score = 255.0 - v.astype(np.float32)
    hsv_score = 0.6 * saturation_score + 0.4 * value_score
    
    # 综合所有方法，取最大值
    combined_score = np.maximum.reduce([red_score, blue_score, purple_score, hsv_score])
    
    # 自适应阈值
    threshold = np.percentile(combined_score, 90)
    mask = (combined_score > threshold).astype(np.uint8)
    
    return _postprocess_mask(mask)


def _postprocess_mask(mask):
    """
    掩膜后处理
    - 形态学闭运算
    - 填充孔洞
    - 保留最大连通域
    """
    # 形态学操作
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    
    # 去除小连通域，保留最大的
    try:
        from skimage import measure
        labeled = measure.label(mask)
        regions = measure.regionprops(labeled)
        if len(regions) > 0:
            # 保留最大的连通域
            largest_region = max(regions, key=lambda r: r.area)
            mask = (labeled == largest_region.label).astype(np.uint8)
    except ImportError:
        # 如果没有skimage，跳过连通域分析
        pass
    
    return mask.astype(np.uint8)


if __name__ == "__main__":
    # 测试
    import matplotlib.pyplot as plt
    
    # 创建测试图像（红色圆形印章）
    size = 256
    image = np.ones((size, size, 3), dtype=np.uint8) * 255
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 3
    circle_mask = (x - center)**2 + (y - center)**2 <= radius**2
    image[circle_mask] = [200, 50, 50]  # 红色
    
    # 提取掩膜
    mask = extract_seal_mask(image, method='multi_color')
    
    print(f"掩膜形状: {mask.shape}")
    print(f"掩膜像素数: {mask.sum()}")
    print(f"掩膜覆盖率: {mask.sum() / mask.size * 100:.2f}%")
