"""
评估指标
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    计算各种评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选，用于计算AUC）
        
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # AUC
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # 真阳性、假阳性、真阴性、假阴性
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        
        # 特异性
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def print_metrics(metrics, prefix=''):
    """打印指标"""
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score:  {metrics['f1']:.4f}")
    
    if 'auc' in metrics and metrics['auc'] > 0:
        print(f"{prefix}AUC:       {metrics['auc']:.4f}")
    
    if 'confusion_matrix' in metrics:
        print(f"{prefix}Confusion Matrix:")
        print(metrics['confusion_matrix'])


def get_classification_report(y_true, y_pred, target_names=None):
    """获取分类报告"""
    if target_names is None:
        target_names = ['Real', 'Fake']
    
    return classification_report(y_true, y_pred, target_names=target_names)
