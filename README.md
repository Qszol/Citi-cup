# 印章真伪鉴别系统 - 完整技术文档

基于多分支深度学习的印章真伪鉴别系统，整合CNN全局特征、物理缺陷检测和几何形变分析三个维度，实现高准确率的印章真伪鉴定。

## 目录

- [系统概述](#系统概述)
- [技术架构](#技术架构)
- [分支详解](#分支详解)
- [训练结果](#训练结果)
- [可视化分析](#可视化分析)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [使用说明](#使用说明)
- [项目结构](#项目结构)

## 系统概述

### 核心思想

印章伪造检测是一个多维度的问题，不同的伪造手段会在不同维度留下痕迹：

- **扫描重印**：边缘模糊、颜色失真、纹理异常
- **手工描摹**：几何形变、笔画不连续
- **高仿私刻**：细微的全局特征差异
- **数字拼接**：局部不一致、频域异常

单一特征难以应对所有伪造类型，因此本系统采用多分支架构，从不同维度提取特征并融合决策。

### 设计原则

1. **互补性**：CNN捕捉全局深度特征，物理分支检测微观缺陷，几何分支分析形状变形
2. **鲁棒性**：多分支投票机制，单一分支失效不影响整体判断
3. **可解释性**：物理和几何特征具有明确的物理意义，便于分析和调试
4. **端到端**：联合训练优化，各分支相互协作

## 技术架构

### 整体架构图

```
输入图像 (224×224×3)
    │
    ├─────────────────┬─────────────────┬─────────────────┐
    │                 │                 │                 │
    ▼                 ▼                 ▼
CNN分支          物理分支          几何分支          
ResNet-50        特征提取          特征提取          
ImageNet预训练   边缘/颜色/纹理    极坐标/网格/环形   
    │                 │                 │                
    ▼                 ▼                 ▼                
2048维特征       33维特征         128维特征        
    │                 │                 │                
    └─────────────────┴─────────────────┘
                            │
                            ▼
                    特征拼接 (2209维)
                            │
                            ▼
                    全连接层 (512维)
                            │
                            ▼
                    全连接层 (256维)
                            │
                            ▼
                    输出层 (2维)
                            │
                            ▼
                    Softmax (真假概率)
```

### 分支特征维度

| 分支 | 特征维度 | 特征类型 | 适用场景 |
|-----|---------|---------|---------|
| CNN | 2048 | 深度特征 | 全局结构、高仿私刻 |
| 物理 | 33 | 手工特征 | 扫描重印、打印伪造 |
| 几何 | 128 | 几何特征 | 手工描摹、形变 |
| 融合 | 2209 | 混合特征 | 所有类型 |

## 分支详解

### 1. CNN主分支

#### 架构设计

```python
输入: 224×224×3 RGB图像
  ↓
ResNet-50骨干网络 (ImageNet预训练)
  ├─ Conv1 + BN + ReLU + MaxPool
  ├─ Layer1 (3个Bottleneck, 256通道)
  ├─ Layer2 (4个Bottleneck, 512通道)
  ├─ Layer3 (6个Bottleneck, 1024通道)
  └─ Layer4 (3个Bottleneck, 2048通道)
  ↓
全局平均池化 (AdaptiveAvgPool2d)
  ↓
2048维特征向量
  ↓
分类头 (Dropout -> FC(512) -> ReLU -> Dropout -> FC(2))
  ↓
输出: 2维logits
```

#### 关键技术

**1. 预训练权重**
- 使用ImageNet预训练的ResNet-50
- 在大规模自然图像上学习的通用特征
- 迁移学习加速收敛，提升性能

**2. 特征提取**
- Layer4输出2048通道的特征图 (7×7×2048)
- 全局平均池化聚合空间信息
- 保留全局结构特征，去除位置敏感性

**3. 分类头设计**
- 两层全连接网络
- Dropout (0.3) 防止过拟合
- ReLU激活增加非线性

#### 可选增强：注意力机制

```python
class CNNBranchWithAttention:
    通道注意力 (Channel Attention)
      - 全局平均池化 + 全局最大池化
      - MLP (2048 -> 128 -> 2048)
      - Sigmoid激活
      - 加权原特征图
    
    空间注意力 (Spatial Attention)
      - 通道维度的平均和最大
      - 7×7卷积
      - Sigmoid激活
      - 加权原特征图
```

**优势**：
- 自动关注印章的关键区域（文字、边缘、五角星等）
- 抑制背景噪声
- 提升对细微差异的敏感性

**代码**：`models/cnn_branch.py`

---

### 2. 物理缺陷分支

#### 特征设计

物理缺陷分支提取33维手工特征，捕捉伪造印章的物理痕迹：

| 特征类型 | 维度 | 计算方法 | 物理意义 |
|---------|------|---------|---------|
| **边缘特征** | 2 | Canny边缘密度、Laplacian方差 | 扫描/打印导致边缘模糊 |
| **颜色特征** | 9 | HSV三通道均值/方差/偏度 | 印泥颜色vs打印颜色差异 |
| **LBP纹理** | 10 | 局部二值模式直方图 | 微观像素颗粒结构 |
| **GLCM纹理** | 5 | 灰度共生矩阵统计值 | 宏观印刷网纹 |
| **Hu矩** | 7 | 7个不变矩 | 轮廓形状不规则度 |

#### 特征提取流程

**1. 边缘特征**
```python
# 边缘密度
edges = cv2.Canny(gray, 50, 150)
edge_density = np.sum(edges > 0) / edges.size

# 拉普拉斯方差（清晰度）
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
```

**2. 颜色特征**
```python
# 转换到HSV空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 提取统计值
for channel in [H, S, V]:
    mean = np.mean(channel)
    std = np.std(channel)
    skewness = scipy.stats.skew(channel.flatten())
```

**3. LBP纹理**
```python
# 局部二值模式
lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')

# 直方图统计
hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
hist = hist / hist.sum()  # 归一化
```

**4. GLCM纹理**
```python
# 灰度共生矩阵
glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256)

# 提取统计值
contrast = graycoprops(glcm, 'contrast')[0, 0]
dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]
```

**5. Hu矩**
```python
# 提取轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算矩
moments = cv2.moments(contours[0])
hu_moments = cv2.HuMoments(moments).flatten()
hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
```

#### 分类器：RBF-SVM

```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',           # 径向基核函数
    C=1.0,                  # 正则化参数
    gamma='scale',          # 核系数
    probability=True,       # 启用概率估计（Platt Scaling）
    class_weight='balanced' # 处理类别不平衡
)
```

**优势**：
- 小样本学习能力强
- 非线性分类边界
- 概率输出便于融合

**代码**：`models/physical_branch.py`

---

### 3. 几何缺陷分支

#### 特征设计

几何分支提取128维特征，分析印章的形状和空间分布。

| 特征类型 | 维度 | 描述 | 检测目标 |
|---------|------|------|---------|
| **极坐标特征** | 36 | 径向3×角度12分bin | 径向分布不均 |
| **网格纹理** | 64 | 4×4和8×8多尺度网格 | 局部密度差异 |
| **环形差异** | 28 | 3个环形区域对比 | 内外圈不一致 |

#### 特征提取流程

**1. 印章归一化**
```python
# 椭圆拟合
ellipse = cv2.fitEllipse(contours)
center, axes, angle = ellipse

# 旋转对齐
M = cv2.getRotationMatrix2D(center, angle, 1.0)
aligned = cv2.warpAffine(mask, M, (w, h))

# 尺度归一化
scale = target_size / max(axes)
normalized = cv2.resize(aligned, None, fx=scale, fy=scale)
```

**2. 极坐标特征**
```python
# 转换到极坐标
for r_bin in [0-33%, 33-66%, 66-100%]:  # 3个径向区间
    for theta_bin in range(12):          # 12个角度区间 (30°)
        # 统计该区域的像素密度
        density = np.sum(mask_region) / region_area
        features.append(density)
```

**3. 网格纹理**
```python
# 4×4网格
for i in range(4):
    for j in range(4):
        grid_region = mask[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
        density = np.sum(grid_region) / grid_region.size
        features.append(density)

# 8×8网格（更细粒度）
# 同样的方式提取 64 维特征
```

**4. 环形差异**
```python
# 定义3个环形区域
rings = [
    (0, r/3),      # 内圈
    (r/3, 2*r/3),  # 中圈
    (2*r/3, r)     # 外圈
]

# 提取每个环的统计特征
for ring in rings:
    mask_ring = create_ring_mask(ring)
    mean = np.mean(mask_ring)
    std = np.std(mask_ring)
    # ... 其他统计值
```

**代码**：`models/geometric_branch.py`

---

## 模型训练

### 快速开始

直接运行训练脚本，使用三分支融合模型（CNN + 物理 + 几何）：

```bash
python train.py --data-root data --batch-size 64 --epochs 35
```

### 三万级数据集重构（推荐先执行）

当真章位于 data/SEAL、假章位于 data/seal-30000 时，推荐先构建标准目录结构并固定随机种子，避免数据泄露与不可复现：

```bash
python rebuild_dataset.py \
    --real-root data/SEAL \
    --fake-root data/seal-30000 \
    --output-root data/seal-30000-split \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42 \
    --clear-output
```

### 推荐训练方式（优先）

优先使用一次性独立划分（train/val/test），其中真章按公司主体分组隔离，假章随机分层，训练阶段默认使用均衡采样：

```bash
python train.py \
    --data-root data/seal-30000-split \
    --batch-size 64 \
    --epochs 35 \
    --lr 3e-4
```

### K 折交叉验证（可选）

如需额外稳定性评估，可在 train+val 上执行分层 K 折，并保持 test 为独立测试集：

```bash
python train.py \
    --data-root data/seal-30000-split \
    --use-kfold \
    --k-folds 5 \
    --batch-size 64 \
    --epochs 35 \
    --lr 3e-4
```

### 主要参数

- `--data-root`: 数据目录（默认 data）
- `--batch-size`: 批次大小（默认 64）
- `--epochs`: 训练轮数（默认 35）
- `--lr`: 学习率（默认 3e-4）
- `--loss`: 损失函数（ce/focal，默认 focal）
- `--optimizer`: 优化器（adam/sgd/adamw，默认 adamw）
- `--early-stop-patience`: 早停patience（默认 10）
- `--balance-train`: 启用训练集类别均衡采样（默认开启）
- `--no-balance-train`: 关闭训练集类别均衡采样
- `--use-kfold`: 是否启用 K 折交叉验证
- `--k-folds`: K 折数（默认 5）
- `--seed`: 随机种子（默认 42）

---

## 模型评估

```bash
python evaluate.py \
    --model-path checkpoints/best_model.pth \
    --data-root data \
    --output-dir evaluation_results
```

评估结果包括：准确率、精确率、召回率、F1分数、AUC、混淆矩阵、ROC曲线、错误样本分析。

**注意**：评估时会自动使用三分支融合模型（与训练时一致）。

## 使用说明

整个鉴别流程已重构整合并接入大语言模型 (LLM) 以输出更专业的评估结果。

### 核心预测脚本

最新的印章查验逻辑已集成于 `visualize_prediction_llm.py` 文件中，执行此脚本可以完成以下过程：
1. 计算出目标的**真章概率**与**假章概率**；
2. 若判断为真章，由大语言模型自动生成一段关于该印章真实性（物理边缘及纹理特性）的总体评价依据；
3. 若判断为假章，将生成异常说明（总体评价），并同时生成并保存一张**注意力热力图**，叠加显示模型聚焦的伪造破绽区域。

### 命令行预测前准备

请确保您已安装必要的深度学习以及 LLM 推理依赖库：
```bash
pip install -r requirements.txt
```

### 模型权重准备
预训练的模型权重文件可从 Hugging Face 公开仓库下载：
[https://huggingface.co/Qszool/Citi-cup-Model/blob/main/best_model.pth]

请在项目目录下创建 `checkpoints/` 目录，并将下载的模型权重文件（`best_model.pth`）放置于 `checkpoints/` 目录下。

### 执行命令

使用大模型鉴别和生成评价（默认通过加载本地 Qwen2.5-0.5B-Instruct 小参数模型进行推断）：
```bash
# 测试指定图片
python visualize_prediction_llm.py --image test_samples/示例图片-real.jpg

# 自定义鉴别权重路径和热力图输出目录
python visualize_prediction_llm.py \
    --image test_samples/示例图片-fake.png \
    --checkpoint checkpoints/best_model.pth \
    --heatmap-dir visualizations/batch_llm
```

### 示例图片判别步骤

项目内已提供两张可直接使用的示例图片：

- `test_samples/示例图片-real.jpg`
- `test_samples/示例图片-fake.png`

请在项目根目录依次执行：

```bash
# 1) 判别 real 示例
F:/conda/python.exe visualize_prediction_llm.py \
    --image test_samples/示例图片-real.jpg \
    --checkpoint checkpoints/best_model.pth \
    --heatmap-dir visualizations/batch_llm

# 2) 判别 fake 示例（会生成热力图）
F:/conda/python.exe visualize_prediction_llm.py \
    --image test_samples/示例图片-fake.png \
    --checkpoint checkpoints/best_model.pth \
    --heatmap-dir visualizations/batch_llm
```

说明：

- real 示例通常不会生成热力图（脚本仅在最终判定为 `fake` 时输出热力图）。
- fake 示例若判定为 `fake`，热力图默认保存在 `visualizations/batch_llm/示例图片-fake_attention_heatmap.png`。

### 输出示例与格式

**终端打印结果：**

```
[>] 开始鉴别印章: data/test/fake/fake_sample_0.png

================ 鉴别结果 ==================
真章概率 (Real Probability) : 0.0117
假章概率 (Fake Probability) : 0.9883
最终判定                   : 假章 (FAKE)
============================================

[>] 正在生成总体评价（判断理由）...

=== 总体评价 ===
该印章在边缘和纹理上的显示已经达到了较高的伪造程度，其密度、锐利度和规则性都明显高于正常印章。模型识别出了一个异常的关注点，即印章的边缘区域出现了一些明显的折痕或不平整的边缘， 这可能是在进行伪造时人为故意造成的。
================

[+] 注意力热力图已保存至: F:\desktop\花旗杯\mywork\真伪\output_attention_heatmap.png
```

**生成的图像文件（假章特有输出）：**
若判别结果为假，程序将会在根目录下生成 `output_attention_heatmap.png` 文件，该文件会通过 **Grad-CAM** 热图叠加标示出图像上异常（即模型认为是造假痕迹）的高亮区域。

## 项目结构

```
seal-authentication/
├── api/                          # API接口与示例
├── checkpoints/                  # 推理权重（best_model.pth）
├── models/                       # 模型定义
├── test_samples/                 # 示例图片（real/fake）
├── utils/                        # 掩膜和工具函数
├── visualize_prediction_llm.py   # 预测、LLM说明与热图生成
├── requirements.txt              # 依赖列表
└── README.md                     # 本文档
```

## 融合策略

### 特征级融合（Feature-level Fusion）

本系统采用特征级融合作为主要策略：

```python
# 特征拼接
cnn_features = cnn_branch(image)          # [B, 2048]
physical_features = physical_branch(image) # [B, 33]
geometric_features = geometric_branch(mask) # [B, 128]

fused_features = torch.cat([
    cnn_features, 
    physical_features, 
    geometric_features
], dim=1)  # [B, 2209]

# 融合分类器
output = fusion_classifier(fused_features)  # [B, 2]
```

#### 融合网络结构

```python
FusionClassifier(
    Linear(2209 -> 512) + BatchNorm + ReLU + Dropout(0.3)
    Linear(512 -> 256) + BatchNorm + ReLU + Dropout(0.3)
    Linear(256 -> 2)
)
```

**优势**：
- 早期融合，各分支特征充分交互
- 端到端训练，联合优化
- 融合层可学习最优特征组合权重

---

## 损失函数

### 1. Focal Loss（主损失）

针对类别不平衡和难分类样本：

```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

其中：
- p_t: 真实类别的预测概率
- α_t: 类别权重（真章0.75，伪章0.8）
- γ: 聚焦参数（默认2.0）
```

**作用**：
- 降低易分类样本的权重
- 聚焦于难分类样本
- 处理真伪样本不平衡

**实现**：`utils/losses.py`

### 2. Center Loss（辅助损失）

度量学习，缩小类内距离：

```python
L_center = (1/2) * Σ ||x_i - c_{y_i}||^2

其中：
- x_i: 样本特征
- c_{y_i}: 类别中心
- y_i: 样本标签
```

**作用**：
- 使同类样本在特征空间聚集
- 增大类间距离
- 提升特征判别性

**权重**：λ_center = 0.1

---

## 训练策略

### 1. 端到端联合训练

```python
# 伪代码
for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播
        cnn_features = cnn_branch(images)
        physical_features = physical_branch(images)
        geometric_features = geometric_branch(masks)
        
        # 特征融合
        fused_features = concat([cnn_features, physical_features, geometric_features])
        logits = fusion_classifier(fused_features)
        
        # 计算损失
        loss = focal_loss(logits, labels) + 0.1 * center_loss(fused_features, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

### 2. 学习率调度

```python
# ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # 监控验证损失
    factor=0.5,        # 学习率衰减因子
    patience=5,        # 容忍轮数
    verbose=True
)
```

### 3. 早停策略

```python
# 监控验证集性能
if val_acc > best_val_acc:
    best_val_acc = val_acc
    save_checkpoint('best_model.pth')
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping")
        break
```

---

## 训练结果

### 数据集信息

- **训练集**：22003张图像
- **验证集**：4714张图像
- **测试集**：4716张图像

### 最终测试集结果

#### 核心指标

| 指标 | 数值 | 评价 |
|-----|------|------|
| **准确率(Accuracy)** | **98.75%** | 优秀 |
| **精确率(Precision)** | **100.00%** | 完美 |
| **召回率(Recall)** | **97.50%** | 优秀 |
| **F1分数 (F1 Score)** | **98.73%** | 优秀 |
| **AUC** | **100.00%** | 完美 |

#### 与单CNN分支对比

| 指标 | 单CNN分支 | 三分支融合 | 提升 |
|-----|----------|-----------|------|
| 准确率 | ~95% | **98.75%** | **+3.75%** |
| F1分数 | ~95% | **98.73%** | **+3.73%** |
| AUC | ~98% | **100%** | **+2%** |

---

## 许可证

MIT License
