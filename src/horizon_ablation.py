"""Compare naive baselines and diagnostic models at 15/30/60-minute horizons.

Run: uv run python src/horizon_ablation.py
Output: results/horizon_ablation/summary.json
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from splits import split_series
from ts_utils import mase, mase_scale, to_serializable

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = REPO_ROOT / "data" / "processed" / "whw_v100.parquet"
OUT_DIR = REPO_ROOT / "results" / "horizon_ablation"
OUT_JSON = OUT_DIR / "summary.json"
TARGET = "target_liters"
HORIZONS_MINUTES = (15, 30, 60)
MINUTE_LAGS = (1, 5, 15, 30, 60, 1440, 10080)
TIMEZONE = "America/Vancouver"
LGBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "random_state": 42,
    "verbosity": -1,
}


def aggregate_target(source: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict]:
    """Sum complete epoch-aligned blocks and return their block-start timestamps."""
    seconds = horizon * 60
    timestamps = source.index.to_numpy(dtype=np.int64)
    # avg_rate at timestamp t is the increment over (t-60s, t]. Assign
    # boundary-ending samples to the block that just ended, not the next one.
    block_starts = ((timestamps - 1) // seconds) * seconds
    grouped = source["avg_rate"].groupby(block_starts).agg(["sum", "size"])
    complete = grouped["size"] == horizon
    blocks = grouped.loc[complete, "sum"].to_frame(TARGET)
    blocks.index = blocks.index.astype(np.int64)
    blocks.index.name = "unix_ts"

    if blocks.empty or not np.all(np.diff(blocks.index.to_numpy()) == seconds):
        raise ValueError(f"{horizon}-minute aggregation is empty or irregular")

    metadata = {
        "alignment": "fixed UTC Unix epoch boundaries; right-closed interval increments",
        "aggregation": "sum of 1-minute avg_rate values; liters consumed per block",
        "input_rows": len(source),
        "complete_blocks": len(blocks),
        "excluded_incomplete_edge_rows": int(source.shape[0] - complete.sum() * horizon),
        "first_block_start_unix_ts": int(blocks.index[0]),
        "last_block_start_unix_ts": int(blocks.index[-1]),
    }
    return blocks, metadata


def add_features(blocks: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[int]]:
    """Add start-time calendar features and exactly representable minute lags."""
    out = blocks.copy()
    local = pd.to_datetime(out.index, unit="s", utc=True).tz_convert(TIMEZONE)
    hour = local.hour
    dow = local.dayofweek
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_local"] = dow
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["weekend"] = dow >= 5

    lag_blocks = [lag // horizon for lag in MINUTE_LAGS if lag % horizon == 0]
    for lag in lag_blocks:
        out[f"lag_{lag}_block"] = out[TARGET].shift(lag)
    out["datetime_local"] = local
    return out, lag_blocks


def _metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scale: float, horizon: int
) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": mae,
        "rmse": rmse,
        "mase": mase(y_true, y_pred, scale),
        "mae_per_minute": mae / horizon,
        "rmse_per_minute": rmse / horizon,
    }


def evaluate_horizon(source: pd.DataFrame, horizon: int) -> dict:
    """Evaluate all Stage 2 baselines and Stage 3 diagnostic models."""
    blocks, aggregation = aggregate_target(source, horizon)
    features, lag_blocks = add_features(blocks, horizon)
    splits = split_series(features)
    train, valid = splits["train"], splits["valid"]
    y_train = train[TARGET].to_numpy(dtype=np.float64)
    y_valid = valid[TARGET].to_numpy(dtype=np.float64)
    scale = mase_scale(y_train)
    mean_train = float(y_train.mean())

    daily_lag = 1440 // horizon
    weekly_lag = 10080 // horizon
    predictions = {
        "always_zero": (np.zeros(len(valid)), "Predict zero liters for every block"),
        "always_mean": (
            np.full(len(valid), mean_train),
            "Predict the training mean block volume",
        ),
        "persistence": (
            valid["lag_1_block"].to_numpy(dtype=np.float64),
            "Predict from the previous completed block",
        ),
        "seasonal_naive_daily": (
            valid[f"lag_{daily_lag}_block"].to_numpy(dtype=np.float64),
            f"Predict from {daily_lag} blocks earlier ({horizon}-minute UTC day)",
        ),
        "seasonal_naive_weekly": (
            valid[f"lag_{weekly_lag}_block"].to_numpy(dtype=np.float64),
            f"Predict from {weekly_lag} blocks earlier ({horizon}-minute UTC week)",
        ),
    }
    baselines = {
        name: {
            "definition": definition,
            **_metrics(y_valid, prediction, scale, horizon),
        }
        for name, (prediction, definition) in predictions.items()
    }

    lag_columns = [f"lag_{lag}_block" for lag in lag_blocks]
    linear_features = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend", *lag_columns
    ]
    tree_features = ["hour", "dow_local", "weekend", *lag_columns]
    models = {}
    for name, model, model_features, params in (
        (
            "linear_regression",
            LinearRegression(),
            linear_features,
            "sklearn LinearRegression, default parameters",
        ),
        (
            "lightgbm",
            lgb.LGBMRegressor(**LGBM_PARAMS),
            tree_features,
            LGBM_PARAMS,
        ),
    ):
        X_train = train[model_features].astype(np.float64).to_numpy()
        X_valid = valid[model_features].astype(np.float64).to_numpy()
        model.fit(X_train, y_train)
        models[name] = {
            "features": model_features,
            "params": params,
            **_metrics(y_valid, model.predict(X_valid), scale, horizon),
        }

    return {
        "horizon_minutes": horizon,
        "target": "sum(avg_rate) over block; liters per block",
        "zero_target_proportion_train": float((train[TARGET] == 0).mean()),
        "aggregation": aggregation,
        "lag_blocks": lag_blocks,
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "validation": {
            "n_observations": len(valid),
            "start_unix_ts": int(valid.index[0]),
            "end_unix_ts": int(valid.index[-1]),
        },
        "mase": {
            "definition": "validation MAE divided by training mean absolute one-block target difference",
            "denominator": scale,
            "denominator_split": "train",
        },
        "baselines": baselines,
        "models": models,
    }


def main() -> None:
    """Run the horizon ablation and save its reproducible summary."""
    source = pd.read_parquet(IN_PARQUET)
    if source.index.name != "unix_ts" or not source.index.is_monotonic_increasing:
        raise ValueError("V100 input must have a sorted unix_ts index")
    if not (source.index.to_series().diff().dropna() == 60).all():
        raise ValueError("V100 input must be a complete one-minute grid")

    summary = {
        "input": "data/processed/whw_v100.parquet",
        "evaluation_split": "validation",
        "split_protocol": (
            "Chronological; validation and test each span 13 local weeks, "
            "with boundaries at Monday 00:00 America/Vancouver"
        ),
        "calendar_timezone": TIMEZONE,
        "seasonal_naive_convention": "fixed UTC-minute lags converted to whole blocks",
        "mase_definition": (
            "validation MAE divided by the training-only mean absolute "
            "one-block difference; computed separately per horizon"
        ),
        "horizons": {
            str(horizon): evaluate_horizon(source, horizon)
            for horizon in HORIZONS_MINUTES
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as output:
        json.dump(to_serializable(summary), output, indent=2, allow_nan=False)
        output.write("\n")

    print(f"Horizon ablation metrics saved to {OUT_JSON.relative_to(REPO_ROOT)}")
    for horizon, result in summary["horizons"].items():
        print(
            f"{horizon} minutes "
            f"(train zero proportion={result['zero_target_proportion_train']:.4f})"
        )
        for group in ("baselines", "models"):
            for name, metrics in result[group].items():
                print(
                    f"  {name}: MAE={metrics['mae']:.6f}, "
                    f"RMSE={metrics['rmse']:.6f}, MASE={metrics['mase']:.6f}, "
                    f"MAE/min={metrics['mae_per_minute']:.6f}"
                )


if __name__ == "__main__":
    main()
