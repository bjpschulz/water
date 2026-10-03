"""
Stage 4 training-objective ablation (1-minute resolution).

Refits the Stage 3 linear regression and untuned LightGBM with different
training losses -- features, split, and hyperparameters are identical, only
the objective changes -- and evaluates every fit on the validation split with
MAE, RMSE and MASE, plus the share of exactly-zero predictions.

Linear: OLS (L2) vs. median regression (L1, statsmodels QuantReg q=0.5).
LightGBM: regression (L2), regression_l1, huber (LightGBM default alpha),
poisson. Every model is scored on every metric; no objective is paired with
"its" metric in advance (AGENTS.md, Stage 4).

Run: uv run python src/objective_ablation.py
Output: output/objective_ablation/summary.json
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from diagnostic_models import FEATURES_LINEAR, FEATURES_TREE, LGBM_PARAMS, TARGET, _xy
from splits import load_features, split_series
from ts_utils import mase, mase_scale, to_serializable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "objective_ablation"
OUT_JSON = OUT_DIR / "summary.json"
OUT_PREDICTIONS = OUT_DIR / "validation_predictions.parquet"

LGBM_OBJECTIVES = ["regression", "regression_l1", "huber", "poisson"]


def fit_linear(X_train, y_train, X_valid, loss: str) -> np.ndarray:
    """OLS (loss 'l2') or median regression (loss 'l1'); returns validation predictions."""
    if loss == "l2":
        return LinearRegression().fit(X_train, y_train).predict(X_valid)
    fit = sm.QuantReg(y_train, sm.add_constant(X_train)).fit(q=0.5, max_iter=5000)
    return sm.add_constant(X_valid, has_constant="add") @ fit.params


def fit_lgbm(X_train, y_train, X_valid, objective: str) -> np.ndarray:
    """Untuned LightGBM (Stage 3 params) with the given objective."""
    model = lgb.LGBMRegressor(objective=objective, **LGBM_PARAMS)
    return model.fit(X_train, y_train).predict(X_valid)


def score(y_true, y_pred, scale: float) -> dict:
    """MAE, RMSE, MASE and share of zero predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mase": mase(y_true, y_pred, scale),
        "zero_prediction_share": float(np.mean(np.abs(y_pred) < 1e-9)),
    }


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Fit every (model, objective) pair and score it on the validation split."""
    train, valid = splits["train"], splits["valid"]
    scale = mase_scale(train[TARGET].to_numpy(dtype=np.float64))
    results, predictions = {}, {}

    Xl_train, y_train = _xy(train, FEATURES_LINEAR)
    Xl_valid, y_valid = _xy(valid, FEATURES_LINEAR)
    for loss in ("l2", "l1"):
        name = f"linear_{loss}"
        predictions[name] = fit_linear(Xl_train, y_train, Xl_valid, loss)
        results[name] = {"features": FEATURES_LINEAR, "objective": loss}

    Xt_train, _ = _xy(train, FEATURES_TREE)
    Xt_valid, _ = _xy(valid, FEATURES_TREE)
    for objective in LGBM_OBJECTIVES:
        name = f"lightgbm_{objective}"
        predictions[name] = fit_lgbm(Xt_train, y_train, Xt_valid, objective)
        results[name] = {
            "features": FEATURES_TREE,
            "objective": objective,
            "params": LGBM_PARAMS,
        }

    for name, y_pred in predictions.items():
        results[name].update(score(y_valid, y_pred, scale))

    pred_frame = pd.DataFrame(
        {
            "unix_ts": valid.index.to_numpy(dtype=np.int64),
            "actual": y_valid,
            **predictions,
        }
    )
    summary = {
        "purpose": (
            "Training-objective ablation: same features, split and "
            "hyperparameters as Stage 3; only the loss changes. See "
            "AGENTS.md Stage 4."
        ),
        "target": TARGET,
        "resolution": "1-minute",
        "evaluation_split": "validation",
        "input_feature_table": "data/processed/whw_features_v1.parquet",
        "baseline_summary_for_comparison": "output/baselines/summary.json",
        "validation_predictions_file": "output/objective_ablation/validation_predictions.parquet",
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "mase": {
            "definition": "validation MAE divided by training mean absolute one-step target difference",
            "denominator": scale,
            "denominator_split": "train",
        },
        "models": results,
    }
    return summary, pred_frame


def main() -> None:
    """Run and save reproducible summary."""
    summary, predictions = run(split_series(load_features()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, allow_nan=False)
        f.write("\n")
    predictions.to_parquet(OUT_PREDICTIONS, index=False)
    print(f"Saved {OUT_JSON.relative_to(REPO_ROOT)}")
    for name, m in summary["models"].items():
        print(
            f"{name:22s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
            f"MASE={m['mase']:.4f} zero_pred={m['zero_prediction_share']:.3f}"
        )


if __name__ == "__main__":
    main()
