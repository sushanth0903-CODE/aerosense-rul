import numpy as np

from src.evaluate import combined_metrics, nasa_asymmetric_score


def test_perfect_prediction_scores_zero():
    y = [0, 10, 50, 100]
    assert nasa_asymmetric_score(y, y) == 0.0


def test_metrics_are_reasonable():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    metrics = combined_metrics(y_true, y_pred)
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["r2"] <= 1
    assert metrics["nasa_score"] >= 0
