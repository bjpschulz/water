"""Evaluate Stage 2 naive baselines on the chronological validation split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from splits import load_features, split_series
from ts_utils import write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "baselines"
OUT_JSON = OUT_DIR / "summary.json"
OUT_PREDICTIONS = OUT_DIR / "validation_predictions.parquet"
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


def calculate_baselines(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict, pd.DataFrame]:
    """Calculate baseline metrics and return their validation predictions."""
    train = splits["train"]
    valid = splits["valid"]
    y_train = train[TARGET].to_numpy(dtype=np.float64)
    y_valid = valid[TARGET].to_numpy(dtype=np.float64)
    train_mean = float(y_train.mean())

    predictions = {
        "always_zero": (np.zeros(len(valid)), "Predict 0 L/min for every row"),
        "always_mean": (
            np.full(len(valid), train_mean),
            "Predict the mean avg_rate from the training split",
        ),
    }
    for name, (col, definition) in LAG_COLUMNS.items():
        predictions[name] = (valid[col].to_numpy(dtype=np.float64), definition)

    pred_frame = pd.DataFrame(
        {
            "unix_ts": valid.index.to_numpy(dtype=np.int64),
            "actual": y_valid,
            **{name: y_pred for name, (y_pred, _) in predictions.items()},
        }
    )

    results = {
        name: {
            "mae": float(mean_absolute_error(y_valid, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
        }
        for name, (y_pred, _) in predictions.items()
    }
    return {
        "split": "validation",
        "train_target_mean": train_mean,
        "baselines": results,
    }, pred_frame


def main() -> None:
    """Run Stage 2 and save its reproducible summary."""
    splits = split_series(load_features())
    summary, predictions = calculate_baselines(splits)
    write_summary(OUT_JSON, summary)
    predictions.to_parquet(OUT_PREDICTIONS, index=False)
    print(f"Saved {OUT_JSON.relative_to(REPO_ROOT)}")
    for name, metrics in summary["baselines"].items():
        print(
            f"{name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}"
        )


if __name__ == "__main__":
    main()
