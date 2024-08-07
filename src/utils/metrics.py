"""
Evaluation metrics for neural networks.
"""
import numpy as np


def accuracy(y_true, y_pred):
    """Calculate accuracy for classification tasks."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average='macro'):
    """Calculate precision score."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(y_true)
    precisions = []
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        
        if tp + fp == 0:
            precisions.append(0.0)
        else:
            precisions.append(tp / (tp + fp))
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
        fp_total = np.sum([np.sum((y_true != cls) & (y_pred == cls)) for cls in classes])
        return tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
    else:
        return precisions


def recall(y_true, y_pred, average='macro'):
    """Calculate recall score."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(y_true)
    recalls = []
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        if tp + fn == 0:
            recalls.append(0.0)
        else:
            recalls.append(tp / (tp + fn))
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
        fn_total = np.sum([np.sum((y_true == cls) & (y_pred != cls)) for cls in classes])
        return tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
    else:
        return recalls


def f1_score(y_true, y_pred, average='macro'):
    """Calculate F1 score."""
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    
    if isinstance(p, list) and isinstance(r, list):
        f1_scores = []
        for pi, ri in zip(p, r):
            if pi + ri == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * pi * ri / (pi + ri))
        return f1_scores
    else:
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error for regression tasks."""
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """Calculate mean absolute error for regression tasks."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Calculate R-squared score for regression tasks."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)
