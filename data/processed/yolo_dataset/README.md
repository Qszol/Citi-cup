# 印章检测YOLOv5数据集

## 数据集概述

本数据集用于训练YOLOv5模型进行印章检测,包含真章和假章两类样本。

## 目录结构

```
yolo_dataset/
├── images/
│   ├── train/          # 训练集图片 (965张)
│   ├── val/            # 验证集图片 (243张)
│   └── test/           # 测试集图片 (474张)
├── labels/
│   ├── train/          # 训练集标签 (965个)
│   ├── val/            # 验证集标签 (243个)
│   └── test/           # 测试集标签 (474个)
├── seal_dataset.yaml   # YOLOv5配置文件
└── README.md           # 本文件
```

## 数据集统计

### 总体统计
- **总样本数**: 1682张
- **训练集**: 965张 (57.4%)
- **验证集**: 243张 (14.4%)
- **测试集**: 474张 (28.2%)

### 真假章分布

| 数据集 | 真章数量 | 假章数量 | 总计 |
|--------|---------|---------|------|
| 训练集 | 793     | 172     | 965  |
| 验证集 | 200     | 43      | 243  |
| 测试集 | 389     | 85      | 474  |
| **总计** | **1382** | **300** | **1682** |

### 数据来源
- **真章**: 来自SEAL2021数据集,已按原始train/val/test划分
- **假章**: 300张合同假章图片,按照真章比例分配到各数据集

## 标注格式

采用YOLO格式标注:
```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 类别ID (0表示印章)
- `x_center, y_center`: 边界框中心点坐标(归一化到0-1)
- `width, height`: 边界框宽高(归一化到0-1)

示例:
```
0 0.803268 0.182493 0.174510 0.239320
```

## 使用方法

### 1. 训练模型

```bash
# 使用YOLOv5s模型
python train.py --data yolo_dataset/seal_dataset.yaml --weights yolov5s.pt --epochs 100

# 使用YOLOv5m模型
python train.py --data yolo_dataset/seal_dataset.yaml --weights yolov5m.pt --epochs 100

# 自定义参数
python train.py --data yolo_dataset/seal_dataset.yaml \
                --weights yolov5s.pt \
                --epochs 100 \
                --batch-size 16 \
                --img-size 640
```

### 2. 验证模型

```bash
python val.py --data yolo_dataset/seal_dataset.yaml --weights runs/train/exp/weights/best.pt
```

### 3. 测试模型

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source yolo_dataset/images/test/
```

## 数据集特点

1. **平衡的数据分布**: 假章按照真章的比例均匀分配到训练、验证和测试集
2. **标准YOLO格式**: 完全符合YOLOv5训练要求
3. **单类别检测**: 专注于印章检测任务
4. **高质量标注**: 所有样本都有对应的边界框标注

## 注意事项

1. 训练前请确保已安装YOLOv5环境
2. 建议使用GPU进行训练以加快速度
3. 可根据实际情况调整batch-size和img-size参数
4. 训练过程中注意观察验证集性能,避免过拟合

## 文件命名规则

- **真章图片**: `cropped_SEAL2021_{train|val|test}_XXXXXXXX.png`
- **假章图片**: `XXXX_contract_with_XXXXXXXX.png`
- **标签文件**: 与图片文件同名,扩展名为`.txt`

## 数据集版本

- **创建日期**: 2026-01-27
- **版本**: 1.0
- **组织工具**: organize_yolo_dataset.py
