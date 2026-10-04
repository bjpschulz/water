"""
Stage 4 feature-group ablation (1-minute resolution).

Refits LightGBM on feature sets built from five blocks and scores it on the
validation split with MAE and RMSE, plus the share of exactly-zero predictions.
Blocks: calendar (hour, day-of-week, weekend), immediate lags (1-15),
hour-scale lags (30, 45, 60), daily lag (1440), weekly lag (10080).

Feature sets (AGENTS.md, Stage 4+):
  calendar_only, lags_only, calendar_immediate, calendar_all, and
  calendar_sparse_short (lags 1, 5, 15 instead of the dense immediate block).

Everything but the feature set is held fixed: same train rows (the lag warm-up
rows are excluded for every set), same split, hyperparameters and seed as
Stage 3. Objectives are L2 and L1; the choice of a main objective is still
open, so no objective is preferred here. Linear regression is not part of
this ablation.

Run: uv run python src/feature_ablation.py
Output: output/feature_ablation/summary.json
"""

from pathlib import Path

import numpy as np
import pandas as pd

from objective_ablation import LAG_COLS, _xy, fit_lgbm, score
from splits import load_features, split_series
from ts_utils import write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "output" / "feature_ablation" / "summary.json"

TARGET = "avg_rate"

CALENDAR = ["hour", "dow_local", "weekend"]
LAG_BLOCKS = {
    "immediate": [f"lag_{k}" for k in range(1, 16)],
    "hour_scale": ["lag_30", "lag_45", "lag_60"],
    "daily": ["lag_1440"],
    "weekly": ["lag_10080"],
}
SPARSE_SHORT = ["lag_1", "lag_5", "lag_15"]

# Feature set -> (include calendar, lag columns).
FEATURE_SETS = {
    "calendar_only": (True, []),
    "lags_only": (False, LAG_COLS),
    "calendar_immediate": (True, LAG_BLOCKS["immediate"]),
    "calendar_all": (True, LAG_COLS),
    "calendar_sparse_short": (True, SPARSE_SHORT),
}

LGBM_OBJECTIVES = ["regression", "regression_l1"]


def feature_columns(calendar: bool, lags: list[str]) -> list[str]:
    """Calendar columns (if included) followed by the lag columns."""
    return (CALENDAR if calendar else []) + lags


def run(splits: dict[str, pd.DataFrame]) -> dict:
    """Fit every (model, feature set) pair and score it on the validation split."""
    train, valid = splits["train"], splits["valid"]
    results = {}
    for objective in LGBM_OBJECTIVES:
        model_name = f"lightgbm_{objective}"
        results[model_name] = {}
        for set_name, (calendar, lags) in FEATURE_SETS.items():
            columns = feature_columns(calendar, lags)
            X_train, y_train = _xy(train, columns)
            X_valid, y_valid = _xy(valid, columns)
            y_pred = fit_lgbm(X_train, y_train, X_valid, objective)
            results[model_name][set_name] = {
                "n_features": len(columns),
                **score(y_valid, y_pred),
            }
            print(
                f"{model_name:24s} {set_name:22s} "
                f"MAE={results[model_name][set_name]['mae']:.4f} "
                f"RMSE={results[model_name][set_name]['rmse']:.4f}"
            )
    return {"split": "validation", "models": results}


def main() -> None:
    """Run and save reproducible summary."""
    summary = run(split_series(load_features()))
    write_summary(OUT_JSON, summary)
    print(f"Saved {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
