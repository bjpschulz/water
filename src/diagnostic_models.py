"""
Fit the Stage 3 diagnostic models -- a plain linear regression and one
quick, untuned LightGBM -- on calendar + full lag-set features, and
evaluate MAE/RMSE on the same chronological validation split used for the
Stage 2 naive baselines (results/baselines/summary.json).

Purpose (AGENTS.md, Stage 3): test whether these features let a model
clear the naive/seasonal-naive floor by a real margin. This is a
diagnostic, not a final tuned model -- no hyperparameter search, no
feature-importance analysis, no residual ACF check (all deferred to
Stage 5+ once Stage 4's resolution decision is made).

Run: uv run python src/diagnostic_models.py
Output: results/diagnostic_models/summary.json
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from splits import load_features, split_series
from ts_utils import to_serializable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "diagnostic_models"
OUT_JSON = OUT_DIR / "summary.json"
TARGET = "avg_rate"

# Full 1-minute lag set from AGENTS.md / build_features.py, shared by both models.
LAG_COLS = ["lag_1", "lag_5", "lag_15", "lag_30", "lag_60", "lag_1440", "lag_10080"]

# Encoding follows the convention already recorded in build_features.py /
# STATE.md: cyclic sin/cos pairs for the linear model, raw integers for
# the tree model (trees split on thresholds and don't need cyclic
# encoding; a linear model needs it to avoid an hour-23-to-hour-0 jump).
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


def fit_and_evaluate(splits: dict[str, pd.DataFrame]) -> dict:
    """Fit each diagnostic model on train, evaluate MAE/RMSE on validation."""
    train, valid = splits["train"], splits["valid"]
    models = {}

    X_train, y_train = _xy(train, FEATURES_LINEAR)
    X_valid, y_valid = _xy(valid, FEATURES_LINEAR)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_valid)
    models["linear_regression"] = {
        "features": FEATURES_LINEAR,
        "params": "sklearn LinearRegression, default parameters",
        "mae": float(mean_absolute_error(y_valid, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
    }

    X_train, y_train = _xy(train, FEATURES_TREE)
    X_valid, y_valid = _xy(valid, FEATURES_TREE)
    gbm = lgb.LGBMRegressor(**LGBM_PARAMS)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_valid)
    models["lightgbm"] = {
        "features": FEATURES_TREE,
        "params": LGBM_PARAMS,
        "mae": float(mean_absolute_error(y_valid, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
    }

    return {
        "stage": 3,
        "purpose": (
            "Diagnostic check: do calendar + lag features let a simple model "
            "clear the naive/seasonal-naive floor by a real margin. Not a "
            "final, tuned model -- see AGENTS.md Stage 3/4."
        ),
        "target": TARGET,
        "resolution": "1-minute",
        "evaluation_split": "validation",
        "input_feature_table": "data/processed/whw_features_v1.parquet",
        "baseline_summary_for_comparison": "results/baselines/summary.json",
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "validation": {
            "n_observations": len(valid),
            "start_unix_ts": int(valid.index[0]),
            "end_unix_ts": int(valid.index[-1]),
        },
        "models": models,
    }


def main() -> None:
    """Run Stage 3 and save its reproducible summary."""
    splits = split_series(load_features())
    summary = fit_and_evaluate(splits)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Diagnostic model metrics saved to {OUT_JSON.relative_to(REPO_ROOT)}")
    for name, metrics in summary["models"].items():
        print(f"{name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")


if __name__ == "__main__":
    main()