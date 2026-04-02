"""
物理缺陷分支模型
提取物理显式特征 + SVM分类
"""

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class PhysicalBranch:
    """
    物理缺陷分支：捕捉高仿印章的微小物理缺陷
    
    特征：
    - 边缘特征 (2维)：边缘密度、拉普拉斯方差
    - 颜色特征 (9维)：HSV三通道的均值、方差、偏度
    - LBP纹理 (10维)：捕捉微观像素颗粒
    - GLCM纹理 (5维)：捕捉宏观印刷网纹
    - Hu矩特征 (7维)：捕捉印章轮廓的不规则度
    
    总计：33维显式特征
    """
    
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.scaler = StandardScaler()
        self.svm_model = None
        self.feature_dim = 33
    
    def extract_features(self, roi_bgr):
        """
        提取物理显式特征
        
        Args:
            roi_bgr: BGR格式的印章ROI图像
            
        Returns:
            features: 33维特征向量
        """
        # 调整大小
        roi_resized = cv2.resize(roi_bgr, (self.img_size, self.img_size))
        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        
        # 提取各类特征
        feat_edge = self._extract_edge_features(roi_gray)
        feat_color = self._extract_color_features(roi_resized)
        feat_lbp = self._extract_lbp_features(roi_gray)
        feat_glcm = self._extract_glcm_features(roi_gray)
        feat_hu = self._extract_hu_features(roi_resized)
        
        # 组合特征
        features = np.array(feat_edge + feat_color + feat_lbp + feat_glcm + feat_hu)
        return features
    
    def _extract_edge_features(self, roi_gray):
        """边缘特征 (2维)：提取边缘密度与拉普拉斯方差"""
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (roi_gray.shape[0] * roi_gray.shape[1])
        laplacian_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        return [edge_density, laplacian_var]
    
    def _extract_color_features(self, roi_bgr):
        """颜色特征 (9维)：提取 HSV 三通道的均值、方差、偏度"""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        moments = []
        for i in range(3):
            pixels = hsv[:, :, i][mask == 255]
            if len(pixels) == 0:
                moments.extend([0.0, 0.0, 0.0])
            else:
                moments.append(float(np.mean(pixels)))
                moments.append(float(np.std(pixels)))
                moments.append(float(skew(pixels)))
        
        return moments
    
    def _extract_lbp_features(self, roi_gray):
        """LBP 纹理 (10维)：捕捉微观像素颗粒"""
        lbp = local_binary_pattern(roi_gray, 8, 1, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist.tolist()
    
    def _extract_glcm_features(self, roi_gray):
        """GLCM 纹理 (5维)：捕捉宏观印刷网纹"""
        glcm = graycomatrix(roi_gray, distances=[1], 
                           angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                           levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        return [float(contrast), float(dissimilarity), float(homogeneity), 
                float(energy), float(correlation)]
    
    def _extract_hu_features(self, roi_bgr):
        """形状与几何特征 (7维)：捕捉印章轮廓的不规则度"""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        moments = cv2.moments(mask)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments.tolist()
    
    def fit(self, X, y, kernel='rbf', C=10.0, gamma='scale'):
        """
        训练SVM模型
        
        Args:
            X: 特征矩阵 [N, 33]
            y: 标签 [N]
            kernel: SVM核函数
            C: 正则化参数
            gamma: 核函数参数
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练SVM
        self.svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        self.svm_model.fit(X_scaled, y)
    
    def predict(self, X):
        """预测"""
        if self.svm_model is None:
            raise ValueError("模型未训练，请先调用fit()")
        
        X_scaled = self.scaler.transform(X)
        return self.svm_model.predict(X_scaled)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.svm_model is None:
            raise ValueError("模型未训练，请先调用fit()")
        
        X_scaled = self.scaler.transform(X)
        return self.svm_model.predict_proba(X_scaled)
    
    def get_feature_names(self):
        """获取特征名称"""
        return (
            ["边缘密度", "边缘锐利度"] +
            ["色相均值", "色相方差", "色相偏度", "饱和度均值", "饱和度方差", "饱和度偏度", 
             "明度均值", "明度方差", "明度偏度"] +
            [f"LBP纹理_Bin{i}" for i in range(10)] +
            ["GLCM_对比度", "GLCM_相异度", "GLCM_同质性", "GLCM_能量", "GLCM_相关性"] +
            [f"Hu几何矩_{i + 1}" for i in range(7)]
        )


if __name__ == "__main__":
    # 测试物理分支
    branch = PhysicalBranch()
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 提取特征
    features = branch.extract_features(test_img)
    print(f"特征维度: {features.shape}")
    print(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")
    
    # 测试训练
    X_train = np.random.randn(100, 33)
    y_train = np.random.randint(0, 2, 100)
    
    branch.fit(X_train, y_train)
    
    # 测试预测
    X_test = np.random.randn(10, 33)
    predictions = branch.predict(X_test)
    probabilities = branch.predict_proba(X_test)
    
    print(f"预测结果: {predictions}")
    print(f"预测概率: {probabilities[0]}")
