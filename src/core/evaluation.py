"""Scoring (MAE, RMSE, zero share) and JSON result summaries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def score(y_true, y_pred) -> dict:
    """MAE, RMSE and the share of exactly-zero predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "zero_prediction_share": float(np.mean(np.abs(y_pred) < 1e-9)),
    }


def to_serializable(obj):
    """Recursively convert numpy/pandas scalars and containers into JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def write_summary(path: Path, summary: dict) -> None:
    """Write a result summary as indented JSON (NaN is rejected)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, allow_nan=False)
        f.write("\n")
