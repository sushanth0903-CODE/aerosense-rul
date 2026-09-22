from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data_pipeline import DEFAULT_DATA_DIR, MAX_RUL, SENSOR_INFO, load_fd001, split_train_validation
from src.evaluate import combined_metrics, save_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": -1,
    "oob_score": True,
}


def fit_model() -> tuple[RandomForestRegressor, dict]:
    """Train Random Forest and save model + metadata + reports."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_fd001(DEFAULT_DATA_DIR)
    fit_df, valid_df = split_train_validation(train_df)

    features = [c for c in fit_df.columns if c not in {"unit_number", "RUL"}]
    variances = fit_df[features].var(numeric_only=True)
    features = [c for c in features if variances[c] > 1e-12]

    X_fit = fit_df[features]
    y_fit = fit_df["RUL"]
    X_valid = valid_df[features]
    y_valid = valid_df["RUL"]

    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    model.fit(X_fit, y_fit)

    valid_pred = np.maximum(model.predict(X_valid), 0)
    validation_metrics = combined_metrics(y_valid, valid_pred)

    # Official C-MAPSS test evaluation uses one RUL target per engine at the
    # final observed cycle, so evaluate only the last row from each test engine.
    final_rows = test_df.loc[test_df.groupby("unit_number")["cycle"].idxmax()].copy()
    X_test = final_rows[features]
    y_test = final_rows["RUL"]
    test_pred = np.maximum(model.predict(X_test), 0)
    test_metrics = combined_metrics(y_test, test_pred)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    top_sensor_features = [
        feature for feature in importance["feature"]
        if feature in SENSOR_INFO
    ][:5]

    feature_ranges = {}
    for feature in features:
        series = fit_df[feature]
        feature_ranges[feature] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }

    metadata = {
        "features": features,
        "top_sensor_features": top_sensor_features,
        "sensor_info": {name: SENSOR_INFO[name] for name in top_sensor_features},
        "feature_ranges": feature_ranges,
        "max_rul_training_target": MAX_RUL,
        "random_forest_params": RANDOM_FOREST_PARAMS,
    }

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(metadata, ARTIFACT_DIR / "metadata.joblib")
    save_metrics(
        {
            "validation_engine_count": int(valid_df["unit_number"].nunique()),
            "validation_row_count": int(len(valid_df)),
            "validation": validation_metrics,
            "test_engine_count": int(final_rows["unit_number"].nunique()),
            "test_final_cycle": test_metrics,
        },
        REPORT_DIR / "metrics.json",
    )

    predictions = final_rows[["unit_number", "cycle", "RUL"]].copy()
    predictions["predicted_RUL"] = test_pred
    predictions["absolute_error"] = np.abs(predictions["RUL"] - predictions["predicted_RUL"])
    predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    return model, metadata


def print_report() -> None:
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    print("\n=== AeroSense training complete ===")
    print(f"Validation RMSE: {metrics['validation']['rmse']:.3f}")
    print(f"Validation MAE : {metrics['validation']['mae']:.3f}")
    print(f"Validation R²  : {metrics['validation']['r2']:.3f}")
    print(f"Validation NASA score: {metrics['validation']['nasa_score']:.3f}")
    print("\nOfficial-style FD001 final-cycle test evaluation:")
    print(f"Test RMSE: {metrics['test_final_cycle']['rmse']:.3f}")
    print(f"Test MAE : {metrics['test_final_cycle']['mae']:.3f}")
    print(f"Test R²  : {metrics['test_final_cycle']['r2']:.3f}")
    print(f"Test NASA score: {metrics['test_final_cycle']['nasa_score']:.3f}")


if __name__ == "__main__":
    fit_model()
    print_report()
