"""
Fit the Stage 3 diagnostic models -- a plain linear regression and one
quick, untuned LightGBM -- on calendar + full lag-set features, and
evaluate MAE/RMSE/MASE on the same chronological validation split used for the
Stage 2 naive baselines (output/baselines/summary.json).

Purpose (AGENTS.md, Stage 3): test whether these features let a model
clear the naive/seasonal-naive floor by a real margin. This is a
diagnostic, not a final tuned model -- no hyperparameter search, no
feature-importance analysis, no residual ACF check (all deferred to
Stage 5+ after Stage 4 selected the modeling resolution).

Run: uv run python src/diagnostic_models.py
Output: output/diagnostic_models/summary.json
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from splits import load_features, split_series
from ts_utils import mase, mase_scale, to_serializable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "diagnostic_models"
OUT_JSON = OUT_DIR / "summary.json"
OUT_PREDICTIONS = OUT_DIR / "validation_predictions.parquet"
TARGET = "avg_rate"

# Full 1-minute lag set from AGENTS.md / build_features.py, shared by both models.
LAG_COLS = ["lag_1", "lag_5", "lag_15", "lag_30", "lag_60", "lag_1440", "lag_10080"]

CALENDAR_LINEAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend"]
CALENDAR_TREE = ["hour", "dow_local", "weekend"]

FEATURES_LINEAR = CALENDAR_LINEAR + LAG_COLS
FEATURES_TREE = CALENDAR_TREE + LAG_COLS

# Untuned by design -- LightGBM sklearn-API defaults plus a fixed seed for
# reproducibility. No search over these; that's explicitly out of scope
# for Stage 3 (see module docstring).
LGBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "random_state": 42,
    "verbosity": -1,
}


def _xy(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) as float64 arrays for a given feature list."""
    X = frame[features].astype(np.float64).to_numpy()
    y = frame[TARGET].to_numpy(dtype=np.float64)
    return X, y


def fit_and_evaluate(splits: dict[str, pd.DataFrame], ) -> tuple[dict, pd.DataFrame]:
    """Fit diagnostic models, evaluate them, and return validation predictions."""
    train, valid = splits["train"], splits["valid"]
    models = {}
    predictions = {}
    scale = mase_scale(train[TARGET].to_numpy(dtype=np.float64))

    X_train, y_train = _xy(train, FEATURES_LINEAR)
    X_valid, y_valid = _xy(valid, FEATURES_LINEAR)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_valid)
    predictions["linear_regression"] = y_pred
    models["linear_regression"] = {
        "features": FEATURES_LINEAR,
        "params": "sklearn LinearRegression, default parameters",
        "mae": float(mean_absolute_error(y_valid, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
        "mase": mase(y_valid, y_pred, scale),
    }

    X_train, y_train = _xy(train, FEATURES_TREE)
    X_valid, y_valid = _xy(valid, FEATURES_TREE)
    gbm = lgb.LGBMRegressor(**LGBM_PARAMS)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_valid)
    predictions["lightgbm"] = y_pred
    models["lightgbm"] = {
        "features": FEATURES_TREE,
        "params": LGBM_PARAMS,
        "mae": float(mean_absolute_error(y_valid, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
        "mase": mase(y_valid, y_pred, scale),
    }

    pred_frame = pd.DataFrame(
        {
            "unix_ts": valid.index.to_numpy(dtype=np.int64),
            "actual": valid[TARGET].to_numpy(dtype=np.float64),
            **predictions,
        }
    )

    return {
        "purpose": (
            "Diagnostic check: do calendar + lag features let a simple model "
            "clear the naive/seasonal-naive floor by a real margin. Not a "
            "final, tuned model -- see AGENTS.md Stage 3/4."
        ),
        "target": TARGET,
        "resolution": "1-minute",
        "evaluation_split": "validation",
        "input_feature_table": "data/processed/whw_features_v1.parquet",
        "baseline_summary_for_comparison": "output/baselines/summary.json",
        "validation_predictions_file": "output/diagnostic_models/validation_predictions.parquet",
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "validation": {
            "n_observations": len(valid),
            "start_unix_ts": int(valid.index[0]),
            "end_unix_ts": int(valid.index[-1]),
        },
        "mase": {
            "definition": "validation MAE divided by training mean absolute one-step target difference",
            "denominator": scale,
            "denominator_split": "train",
        },
        "models": models,
    }, pred_frame


def main() -> None:
    """Run and save reproducible summary."""
    splits = split_series(load_features())
    summary, predictions = fit_and_evaluate(splits)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, allow_nan=False)
        f.write("\n")
    predictions.to_parquet(OUT_PREDICTIONS, index=False)
    print(f"Diagnostic model metrics saved to {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Diagnostic model predictions saved to {OUT_PREDICTIONS.relative_to(REPO_ROOT)}")
    for name, metrics in summary["models"].items():
        print(
            f"{name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
            f"MASE={metrics['mase']:.6f}"
        )


if __name__ == "__main__":
    main()
