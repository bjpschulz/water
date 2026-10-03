"""
Stage 3 training-objective ablation (1-minute resolution).

Fits a linear regression and one untuned LightGBM on calendar + full lag-set
features with different training losses -- features, split, and
hyperparameters are identical, only the objective changes -- and evaluates
every fit on the validation split with MAE and RMSE, plus the share of
exactly-zero predictions. The L2 fits are the plain diagnostic models.

Linear: OLS (L2) vs. median regression (L1, statsmodels QuantReg q=0.5).
LightGBM: regression (L2), regression_l1, huber (LightGBM default alpha),
poisson. Every model is scored on every metric; no objective is paired with
"its" metric in advance (AGENTS.md, Stage 3).

Run: uv run python src/objective_ablation.py
Output: output/objective_ablation/summary.json (metrics) and
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
from ts_utils import write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "objective_ablation"
OUT_JSON = OUT_DIR / "summary.json"
OUT_PREDICTIONS = OUT_DIR / "validation_predictions.parquet"

TARGET = "avg_rate"

# Full 1-minute lag set from AGENTS.md / build_features.py.
LAG_COLS = ["lag_1", "lag_5", "lag_15", "lag_30", "lag_60", "lag_1440", "lag_10080"]
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
LGBM_OBJECTIVES = ["regression", "regression_l1", "huber", "poisson"]


def _xy(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) as float64 arrays for a given feature list."""
    X = frame[features].astype(np.float64).to_numpy()
    y = frame[TARGET].to_numpy(dtype=np.float64)
    return X, y


def fit_linear(X_train, y_train, X_valid, loss: str) -> np.ndarray:
    """OLS (loss 'l2') or median regression (loss 'l1'); returns validation predictions."""
    if loss == "l2":
        return LinearRegression().fit(X_train, y_train).predict(X_valid)
    fit = sm.QuantReg(y_train, sm.add_constant(X_train)).fit(q=0.5, max_iter=5000)
    return sm.add_constant(X_valid, has_constant="add") @ fit.params


def fit_lgbm(X_train, y_train, X_valid, objective: str) -> np.ndarray:
    """Untuned LightGBM with the given objective."""
    model = lgb.LGBMRegressor(objective=objective, **LGBM_PARAMS)
    return model.fit(X_train, y_train).predict(X_valid)


def score(y_true, y_pred) -> dict:
    """MAE, RMSE and share of zero predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "zero_prediction_share": float(np.mean(np.abs(y_pred) < 1e-9)),
    }


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Fit every (model, objective) pair and score it on the validation split."""
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
    for objective in LGBM_OBJECTIVES:
        name = f"lightgbm_{objective}"
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
