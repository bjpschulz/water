"""
Stage 6 single test evaluation (1-minute resolution).

The pre-declared models -- LightGBM and the linear benchmark, each under an L2
and an L1 loss, on the final feature set (`hour` + `lag_1`-`lag_15`, raw hour)
-- are fitted on train + valid with the Stage 3 settings, frozen before this
script was written, and scored once on the test split with MAE, RMSE and the
share of exactly-zero predictions, next to the five naive baselines. The
always-mean baseline uses the train + valid mean, i.e. the same data as the
models. Nothing here is tuned or selected on the test split.

Run: uv run python src/test_evaluation.py
Output: output/test_evaluation/summary.json
"""

from pathlib import Path

import numpy as np
import pandas as pd

from baselines import LAG_COLUMNS
from diagnostic_models import LGBM_OBJECTIVES, TARGET, _xy, fit_linear, score, train_lgbm
from feature_importance import FEATURE_SETS
from splits import load_features, split_series
from ts_utils import write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "output" / "test_evaluation" / "summary.json"

FINAL_COLUMNS = FEATURE_SETS["final"][0]


def main() -> None:
    """Fit on train + valid, score every model and baseline once on test, save the summary."""
    splits = split_series(load_features())
    fit_frame = pd.concat([splits["train"], splits["valid"]])  # contiguous, chronological
    test = splits["test"]
    X_fit, y_fit = _xy(fit_frame, FINAL_COLUMNS)
    X_test, y_test = _xy(test, FINAL_COLUMNS)
    fit_mean = float(y_fit.mean())

    predictions = {
        "always_zero": ("baseline", np.zeros(len(test))),
        "always_mean": ("baseline", np.full(len(test), fit_mean)),
        **{name: ("baseline", test[col].to_numpy(dtype=np.float64)) for name, (col, _) in LAG_COLUMNS.items()},
    }
    for loss in ("l2", "l1"):
        predictions[f"linear_{loss}"] = ("model", fit_linear(X_fit, y_fit, X_test, loss))
    for suffix, objective in LGBM_OBJECTIVES.items():
        predictions[f"lightgbm_{suffix}"] = ("model", train_lgbm(X_fit, y_fit, objective).predict(X_test))

    summary = {
        "split": "test",
        "features": FINAL_COLUMNS,
        "fit_rows": len(fit_frame),
        "n": len(test),
        "fit_target_mean": fit_mean,
        "results": {name: {"kind": kind, **score(y_test, y_pred)} for name, (kind, y_pred) in predictions.items()},
    }
    write_summary(OUT_JSON, summary)
    print(f"{'predictor':24s} {'kind':9s} {'MAE':>8s} {'RMSE':>8s} {'zero_pred':>10s}")
    for name, r in summary["results"].items():
        print(f"{name:24s} {r['kind']:9s} {r['mae']:8.4f} {r['rmse']:8.4f} {r['zero_prediction_share']:10.3f}")
    print(f"\nn={summary['n']} fit_rows={summary['fit_rows']}\nSaved {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
