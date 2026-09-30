"""Evaluate Stage 2 naive baselines on the chronological validation split."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from splits import load_features, split_series
from ts_utils import mase, mase_scale, to_serializable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "baselines"
OUT_JSON = OUT_DIR / "summary.json"
TARGET = "avg_rate"

# Lag baselines are read directly from the feature table's own lag_k
# columns (built and self-checked in build_features.py) rather than
# re-derived here -- lag_k[t] == avg_rate[t-k] is already guaranteed for
# every row of the table before it's ever split, so recomputing it on the
# validation slice would just be re-proving an invariant that's true by
# construction.
LAG_COLUMNS = {
    "persistence": ("lag_1", "Predict avg_rate from 1 UTC-minute earlier"),
    "seasonal_naive_daily": ("lag_1440", "Predict avg_rate from 1,440 UTC-minute rows earlier"),
    "seasonal_naive_weekly": ("lag_10080", "Predict avg_rate from 10,080 UTC-minute rows earlier"),
}


def calculate_baselines(splits: dict[str, pd.DataFrame]) -> dict:
    """Calculate MAE, RMSE, and MASE for fixed baselines."""
    train = splits["train"]
    valid = splits["valid"]
    y_train = train[TARGET].to_numpy(dtype=np.float64)
    y_valid = valid[TARGET].to_numpy(dtype=np.float64)
    train_mean = float(y_train.mean())
    scale = mase_scale(y_train)

    predictions = {
        "always_zero": (np.zeros(len(valid)), "Predict 0 L/min for every row"),
        "always_mean": (
            np.full(len(valid), train_mean),
            "Predict the mean avg_rate from the training split",
        ),
    }
    for name, (col, definition) in LAG_COLUMNS.items():
        predictions[name] = (valid[col].to_numpy(dtype=np.float64), definition)

    results = {}
    for name, (y_pred, definition) in predictions.items():
        results[name] = {
            "definition": definition,
            "mae": float(mean_absolute_error(y_valid, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
            "mase": mase(y_valid, y_pred, scale),
        }

    return {
        "target": TARGET,
        "resolution": "1-minute",
        "evaluation_split": "validation",
        "input_feature_table": "data/processed/whw_features_v1.parquet",
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "validation": {
            "n_observations": len(valid),
            "start_unix_ts": int(valid.index[0]),
            "end_unix_ts": int(valid.index[-1]),
        },
        "train_target_mean": train_mean,
        "mase": {
            "definition": "validation MAE divided by training mean absolute one-step target difference",
            "denominator": scale,
            "denominator_split": "train",
        },
        "baselines": results,
    }


def main() -> None:
    """Run Stage 2 and save its reproducible summary."""
    splits = split_series(load_features())
    summary = calculate_baselines(splits)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Baseline metrics saved to {OUT_JSON.relative_to(REPO_ROOT)}")
    for name, metrics in summary["baselines"].items():
        print(
            f"{name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
            f"MASE={metrics['mase']:.6f}"
        )


if __name__ == "__main__":
    main()
