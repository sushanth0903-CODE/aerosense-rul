from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return standard regression metrics."""
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def nasa_asymmetric_score(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Compute the PHM/NASA asymmetric RUL score.

    d = predicted_RUL - actual_RUL
    d < 0: exp(-d/13) - 1
    d >= 0: exp(d/10) - 1
    Lower is better; a perfect predictor scores 0.
    """
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    d = y_pred - y_true

    penalties = np.where(
        d < 0,
        np.exp(np.clip(-d / 13.0, -50, 50)) - 1.0,
        np.exp(np.clip(d / 10.0, -50, 50)) - 1.0,
    )
    return float(np.sum(penalties))


def combined_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return RMSE, MAE, R², and asymmetric score."""
    result = regression_metrics(y_true, y_pred)
    result["nasa_score"] = nasa_asymmetric_score(y_true, y_pred)
    return result


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def health_status(rul: float) -> tuple[str, str]:
    """Return demo UI status and explanation for a predicted RUL."""
    if rul > 60:
        return "SAFE", "More than 60 cycles predicted remaining."
    if rul > 20:
        return "MAINTENANCE WARNING", "Between 20 and 60 cycles predicted remaining."
    return "CRITICAL", "20 cycles or fewer predicted remaining."
