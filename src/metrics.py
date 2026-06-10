from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def safe_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def classification_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = safe_confusion_counts(y_true, y_pred)
    return {
        "accuracy": _safe_divide(tp + tn, len(y_true)),
        "sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "ppv": _safe_divide(tp, tp + fp),
        "npv": _safe_divide(tn, tn + fn),
    }


def sensitivity_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, target_specificity: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    mask = specificity >= target_specificity
    if not np.any(mask):
        return float("nan")
    return float(np.max(tpr[mask]))


def specificity_at_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    mask = tpr >= target_sensitivity
    if not np.any(mask):
        return float("nan")
    return float(np.max(specificity[mask]))


def calibration_curve_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.digitize(y_prob, bins) - 1
    x_vals: list[float] = []
    y_vals: list[float] = []
    for bin_idx in range(n_bins):
        mask = assignments == bin_idx
        if np.any(mask):
            x_vals.append(float(np.mean(y_prob[mask])))
            y_vals.append(float(np.mean(y_true[mask])))
    return {"predicted": x_vals, "observed": y_vals}


def compute_binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "brier_score": float(brier_score_loss(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        metrics["sensitivity_at_90_specificity"] = sensitivity_at_specificity(y_true, y_prob, 0.90)
        metrics["specificity_at_90_sensitivity"] = specificity_at_sensitivity(y_true, y_prob, 0.90)
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
        metrics["sensitivity_at_90_specificity"] = float("nan")
        metrics["specificity_at_90_sensitivity"] = float("nan")

    metrics.update(classification_metrics_at_threshold(y_true, y_prob, threshold=0.5))
    metrics["calibration_curve"] = calibration_curve_points(y_true, y_prob)
    return metrics
