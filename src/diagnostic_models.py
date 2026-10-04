"""
Stage 3 diagnostic models (1-minute resolution).

Fits a linear regression and one untuned LightGBM on calendar + full lag-set
features, each under an L2 and an L1 training loss -- features, split and
hyperparameters are identical, only the loss changes -- and evaluates every fit
on the validation split with MAE and RMSE, plus the share of exactly-zero
predictions. The L2 fits are the plain diagnostic models.

Linear: OLS (L2) vs. median regression (L1, statsmodels QuantReg q=0.5).
LightGBM: regression_l2 vs. regression_l1. Every model is scored on every
metric; no loss is paired with "its" metric in advance (AGENTS.md, Stage 3).

Run: uv run python src/diagnostic_models.py
Output: output/diagnostic_models/summary.json (metrics) and
validation_predictions.parquet (one column per fitted model)
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from splits import load_features, split_series
from ts_utils import LAGS, write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "diagnostic_models"
OUT_JSON = OUT_DIR / "summary.json"
OUT_PREDICTIONS = OUT_DIR / "validation_predictions.parquet"

TARGET = "avg_rate"

# Full 1-minute lag set (ts_utils.LAGS), as built in build_features.py.
LAG_COLS = [f"lag_{k}" for k in LAGS]
FEATURES_LINEAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend"] + LAG_COLS
FEATURES_TREE = ["hour", "dow_local", "weekend"] + LAG_COLS

# Untuned by design: fixed values and seed, no search.
LGBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "random_state": 42,
    "verbosity": -1,
}
# Model-name suffix -> LightGBM objective.
LGBM_OBJECTIVES = {
    "l2": "regression_l2",
    "l1": "regression_l1",
}


def _xy(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) as float64 arrays for a given feature list."""
    X = frame[features].astype(np.float64).to_numpy()
    y = frame[TARGET].to_numpy(dtype=np.float64)
    return X, y


def fit_linear(X_train, y_train, X_valid, loss: str) -> np.ndarray:
    """OLS (loss 'l2') or median regression (loss 'l1'); returns validation predictions."""
    if loss == "l2":
        return LinearRegression().fit(X_train, y_train).predict(X_valid)
    elif loss == "l1":
        fit = sm.QuantReg(y_train, sm.add_constant(X_train)).fit(q=0.5, max_iter=5000)
        return sm.add_constant(X_valid, has_constant="add") @ fit.params


def train_lgbm(X_train, y_train, objective: str) -> lgb.LGBMRegressor:
    """Untuned LightGBM (LGBM_PARAMS) with the given objective, fitted."""
    return lgb.LGBMRegressor(objective=objective, **LGBM_PARAMS).fit(X_train, y_train)


def fit_lgbm(X_train, y_train, X_valid, objective: str) -> np.ndarray:
    """Untuned LightGBM with the given objective; returns validation predictions."""
    return train_lgbm(X_train, y_train, objective).predict(X_valid)


def score(y_true, y_pred) -> dict:
    """MAE, RMSE and share of zero predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "zero_prediction_share": float(np.mean(np.abs(y_pred) < 1e-9)),
    }


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Fit every (model, loss) pair and score it on the validation split."""
    train, valid = splits["train"], splits["valid"]
    results, predictions = {}, {}

    Xl_train, y_train = _xy(train, FEATURES_LINEAR)
    Xl_valid, y_valid = _xy(valid, FEATURES_LINEAR)
    for loss in ("l2", "l1"):
        name = f"linear_{loss}"
        predictions[name] = fit_linear(Xl_train, y_train, Xl_valid, loss)
        results[name] = {"objective": loss}

    Xt_train, _ = _xy(train, FEATURES_TREE)
    Xt_valid, _ = _xy(valid, FEATURES_TREE)
    for suffix, objective in LGBM_OBJECTIVES.items():
        name = f"lightgbm_{suffix}"
        predictions[name] = fit_lgbm(Xt_train, y_train, Xt_valid, objective)
        results[name] = {"objective": objective}

    for name, y_pred in predictions.items():
        results[name].update(score(y_valid, y_pred))

    pred_frame = pd.DataFrame(
        {
            "unix_ts": valid.index.to_numpy(dtype=np.int64),
            "actual": y_valid,
            **predictions,
        }
    )
    summary = {"split": "validation", "models": results}
    return summary, pred_frame


def main() -> None:
    """Run and save reproducible summary."""
    summary, predictions = run(split_series(load_features()))
    write_summary(OUT_JSON, summary)
    predictions.to_parquet(OUT_PREDICTIONS, index=False)
    print(f"Saved {OUT_JSON.relative_to(REPO_ROOT)}")
    for name, m in summary["models"].items():
        print(
            f"{name:22s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
            f"zero_pred={m['zero_prediction_share']:.3f}"
        )


if __name__ == "__main__":
    main()
