"""Metrics for adaptation under shift.

Accuracy alone hides the characteristic TTA failure: entropy minimisation can
collapse the model onto one class and still post a respectable number on an
imbalanced test set. ``collapse_ratio`` and ``balanced_accuracy`` exist to make
that visible, and should be reported alongside accuracy in any result table.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall. Immune to a collapsed majority-class predictor."""
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores)) if scores else 0.0


def collapse_ratio(y_pred: np.ndarray, n_classes: int) -> float:
    """Share of predictions falling in the single most-predicted class.

    1/n_classes means a perfectly balanced predictor; 1.0 means total collapse.
    Any value near 1.0 invalidates the accuracy figure next to it.
    """
    if y_pred.size == 0:
        return 0.0
    counts = np.bincount(y_pred, minlength=n_classes)
    return float(counts.max() / counts.sum())


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """Standard binned ECE over max-probability confidence."""
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Left-open bins, with the first bin closed so confidence==0 lands somewhere.
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        if not mask.any():
            continue
        weight = mask.mean()
        ece += weight * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_metrics(
    y_true: np.ndarray, probs: np.ndarray, n_classes: int | None = None
) -> dict[str, float]:
    """Full metric bundle from ground truth and predicted probabilities."""
    n_classes = n_classes or probs.shape[1]
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "collapse_ratio": collapse_ratio(y_pred, n_classes),
        "ece": expected_calibration_error(y_true, probs),
    }
