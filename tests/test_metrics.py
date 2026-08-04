import numpy as np

from eanet.eval.metrics import (
    balanced_accuracy,
    collapse_ratio,
    compute_metrics,
    expected_calibration_error,
    macro_f1,
)


def test_perfect_predictions():
    y = np.array([0, 1, 0, 1])
    probs = np.eye(2)[y]
    out = compute_metrics(y, probs)
    assert out["accuracy"] == 1.0
    assert out["balanced_accuracy"] == 1.0
    assert out["macro_f1"] == 1.0


def test_collapse_is_detected_where_accuracy_hides_it():
    """A degenerate all-one-class predictor on imbalanced data."""
    y = np.array([0] * 9 + [1])
    y_pred = np.zeros(10, dtype=int)
    probs = np.eye(2)[y_pred]

    out = compute_metrics(y, probs)
    assert out["accuracy"] == 0.9  # looks fine
    assert out["balanced_accuracy"] == 0.5  # is not fine
    assert out["collapse_ratio"] == 1.0


def test_collapse_ratio_floor_is_uniform():
    y_pred = np.array([0, 1, 0, 1])
    assert collapse_ratio(y_pred, 2) == 0.5


def test_balanced_accuracy_ignores_absent_classes():
    y = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    assert balanced_accuracy(y, y_pred, n_classes=3) == 1.0


def test_macro_f1_handles_never_predicted_class():
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    assert 0.0 < macro_f1(y, y_pred, 2) < 1.0


def test_ece_zero_when_perfectly_calibrated():
    y = np.array([0, 1])
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, probs) < 1e-9


def test_ece_large_when_confidently_wrong():
    y = np.array([0, 0])
    probs = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert expected_calibration_error(y, probs) > 0.9


def test_metric_keys():
    y = np.array([0, 1])
    probs = np.eye(2)[y]
    assert set(compute_metrics(y, probs)) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "collapse_ratio",
        "ece",
    }
