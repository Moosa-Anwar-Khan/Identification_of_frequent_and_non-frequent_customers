import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

sns.set(style="whitegrid")


def encode_binary_labels(series: pd.Series) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Encode common project labels into {0,1}. Returns (y, id2label).
    Supports:
      - "Frequent" / "Non-frequent"
      - "Frequent (product)" / "Non-frequent (product)"
    """
    s = series.astype(str).str.strip().str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)

    mapping = {
        "non-frequent": 0,
        "non frequent": 0,
        "frequent": 1,
        "non-frequent (product)": 0,
        "non frequent (product)": 0,
        "frequent (product)": 1,
        "0": 0,
        "1": 1,
    }
    y = s.map(mapping)
    if y.isna().any():
        unknown = sorted(s[y.isna()].unique().tolist())[:10]
        raise ValueError(f"Unknown labels found (showing up to 10): {unknown}")
    y = y.astype(int).values
    id2label = {0: "Non-frequent", 1: "Frequent"}
    return y, id2label


def best_threshold_by_fbeta(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    beta: float = 1.0,
    min_recall: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Return (best_threshold, best_fbeta_for_class_1) based on precision-recall thresholds.
    beta > 1 favors recall; beta < 1 favors precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob_pos)
    # thresholds has len = len(precision)-1
    if len(thresholds) == 0:
        return 0.5, 0.0

    beta = float(beta)
    beta2 = beta * beta
    denom = (beta2 * precision[:-1]) + recall[:-1]
    fbeta = (1.0 + beta2) * precision[:-1] * recall[:-1] / np.clip(denom, 1e-12, None)

    if min_recall is not None:
        mask = recall[:-1] >= float(min_recall)
        if mask.any():
            fbeta_masked = np.where(mask, fbeta, -np.inf)
            best_idx = int(np.nanargmax(fbeta_masked))
            return float(thresholds[best_idx]), float(fbeta[best_idx])

    best_idx = int(np.nanargmax(fbeta))
    return float(thresholds[best_idx]), float(fbeta[best_idx])


def best_threshold_by_f1(y_true: np.ndarray, prob_pos: np.ndarray) -> Tuple[float, float]:
    """
    Return (best_threshold, best_f1_for_class_1) based on precision-recall thresholds.
    """
    return best_threshold_by_fbeta(y_true, prob_pos, beta=1.0, min_recall=None)


def save_classification_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    prefix: str,
    id2label: Optional[Dict[int, str]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if id2label:
        labels = sorted(id2label)
        target_names = [id2label[i] for i in labels]
    else:
        labels = None
        target_names = None

    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    rep_df = pd.DataFrame(rep).transpose()
    rep_df.to_csv(os.path.join(out_dir, f"{prefix}_classification_report.csv"))

    rep_txt = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    with open(os.path.join(out_dir, f"{prefix}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_txt)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = "Confusion matrix",
):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    out_path: str,
    title: str = "Precision-Recall curve",
    best_threshold: Optional[float] = None,
):
    precision, recall, thresholds = precision_recall_curve(y_true, prob_pos)
    ap = average_precision_score(y_true, prob_pos)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    t = f"{title} (AP={ap:.4f})"
    plt.title(t)

    if best_threshold is not None and thresholds is not None and len(thresholds) > 0:
        # Finding the PR point closest to best_threshold
        idx = int(np.argmin(np.abs(thresholds - best_threshold)))
        # thresholds aligns with precision[:-1], recall[:-1]
        plt.scatter([recall[idx]], [precision[idx]])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
