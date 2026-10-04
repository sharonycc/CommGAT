import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)


def _safe_confusion_stats(y_true, y_pred):
    """
    返回 tn, fp, fn, tp
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0, 0, 0, 0
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def _precision_at_k(y_true, y_score, k_ratio):
    """
    取分数最高的前 k_ratio 比例样本，计算其中正样本比例
    """
    n = len(y_true)
    if n == 0:
        return np.nan

    k = max(1, int(np.ceil(n * k_ratio)))
    order = np.argsort(-y_score)   # descending
    top_idx = order[:k]

    top_true = y_true[top_idx]
    precision_k = np.mean(top_true == 1)
    return float(precision_k)


def _fp_rate_at_k(y_true, y_score, k_ratio):
    """
    取分数最高的前 k_ratio 比例样本，计算其中假阳性比例
    即 top-k 中 y_true==0 的比例
    """
    n = len(y_true)
    if n == 0:
        return np.nan

    k = max(1, int(np.ceil(n * k_ratio)))
    order = np.argsort(-y_score)
    top_idx = order[:k]

    top_true = y_true[top_idx]
    fp_rate_k = np.mean(top_true == 0)
    return float(fp_rate_k)


def evaluate_pseudo_edge_scores(scores, labels, threshold=None, prefix=""):
    """
    scores: np.ndarray, shape [num_edges]
    labels: np.ndarray, values in {0,1,-1}
    -1 表示忽略

    返回：
      out: dict
      threshold: float
    """
    valid_idx = np.where(labels >= 0)[0]
    y_true = labels[valid_idx]
    y_score = scores[valid_idx]

    # 如果没有有效样本
    if len(y_true) == 0:
        out = {
            f"{prefix} TN": np.nan,
            f"{prefix} FP": np.nan,
            f"{prefix} FN": np.nan,
            f"{prefix} TP": np.nan,
            f"{prefix} AUROC": np.nan,
            f"{prefix} AP": np.nan,
            f"{prefix} AUPRC": np.nan,
            f"{prefix} ACC": np.nan,
            f"{prefix} Precision": np.nan,
            f"{prefix} Recall": np.nan,
            f"{prefix} F1": np.nan,
            f"{prefix} FPR": np.nan,
            f"{prefix} TNR": np.nan,
            f"{prefix} Precision@1%": np.nan,
            f"{prefix} Precision@5%": np.nan,
            f"{prefix} Precision@10%": np.nan,
            f"{prefix} Top1% FP rate": np.nan,
            f"{prefix} Top5% FP rate": np.nan,
            f"{prefix} Top10% FP rate": np.nan,
        }
        return out, 0.5

    # 若只有单类，AUROC 无法计算
    if len(np.unique(y_true)) < 2:
        out = {
            f"{prefix} TN": np.nan,
            f"{prefix} FP": np.nan,
            f"{prefix} FN": np.nan,
            f"{prefix} TP": np.nan,
            f"{prefix} AUROC": np.nan,
            f"{prefix} AP": np.nan,
            f"{prefix} AUPRC": np.nan,
            f"{prefix} ACC": np.nan,
            f"{prefix} Precision": np.nan,
            f"{prefix} Recall": np.nan,
            f"{prefix} F1": np.nan,
            f"{prefix} FPR": np.nan,
            f"{prefix} TNR": np.nan,
            f"{prefix} Precision@1%": _precision_at_k(y_true, y_score, 0.01),
            f"{prefix} Precision@5%": _precision_at_k(y_true, y_score, 0.05),
            f"{prefix} Precision@10%": _precision_at_k(y_true, y_score, 0.10),
            f"{prefix} Top1% FP rate": _fp_rate_at_k(y_true, y_score, 0.01),
            f"{prefix} Top5% FP rate": _fp_rate_at_k(y_true, y_score, 0.05),
            f"{prefix} Top10% FP rate": _fp_rate_at_k(y_true, y_score, 0.10),
        }
        return out, 0.5

    # 连续分数指标
    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_score)
    auprc = auc(recall_curve, precision_curve)

    # 阈值：仍然默认用 F1 最大
    if threshold is None:
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-12)
        best_idx = np.nanargmax(f1_scores)

        if len(thresholds) == 0:
            threshold = 0.5
        else:
            if best_idx >= len(thresholds):
                best_idx = len(thresholds) - 1
            threshold = thresholds[best_idx]

    # 二值化
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = _safe_confusion_stats(y_true, y_pred)
    fpr = fp / (fp + tn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)

    # Top-k 重点指标
    p_at_1 = _precision_at_k(y_true, y_score, 0.01)
    p_at_5 = _precision_at_k(y_true, y_score, 0.05)
    p_at_10 = _precision_at_k(y_true, y_score, 0.10)

    fp_at_1 = _fp_rate_at_k(y_true, y_score, 0.01)
    fp_at_5 = _fp_rate_at_k(y_true, y_score, 0.05)
    fp_at_10 = _fp_rate_at_k(y_true, y_score, 0.10)

    out = {
        f"{prefix} TN": float(tn),
        f"{prefix} FP": float(fp),
        f"{prefix} FN": float(fn),
        f"{prefix} TP": float(tp),
        f"{prefix} AUROC": float(auroc),
        f"{prefix} AP": float(ap),
        f"{prefix} AUPRC": float(auprc),
        f"{prefix} ACC": float(acc),
        f"{prefix} Precision": float(precision),
        f"{prefix} Recall": float(recall),
        f"{prefix} F1": float(f1),
        f"{prefix} FPR": float(fpr),
        f"{prefix} TNR": float(tnr),
        f"{prefix} Precision@1%": float(p_at_1),
        f"{prefix} Precision@5%": float(p_at_5),
        f"{prefix} Precision@10%": float(p_at_10),
        f"{prefix} Top1% FP rate": float(fp_at_1),
        f"{prefix} Top5% FP rate": float(fp_at_5),
        f"{prefix} Top10% FP rate": float(fp_at_10),
    }

    return out, float(threshold)