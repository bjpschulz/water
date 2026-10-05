"""
Stage 6: single test evaluation.

The pre-declared final models -- LightGBM and the linear benchmark, each under
an L2 and an L1 loss, on the final feature set (`hour` + `lag_1`-`lag_15`, raw
hour) -- are fitted on train + valid with the Stage 3 settings and scored once
on the test split, next to the five naive baselines. The always-mean baseline
uses the train + valid mean, i.e. the same data as the models. Nothing here is
tuned or selected on the test split.

Run: uv run python src/test_evaluation.py
Output: output/test_evaluation/summary.json and test_predictions.parquet
"""

import numpy as np
import pandas as pd

from core.config import FINAL_FEATURES, LOSSES, OUTPUT_DIR
from core.data import load_features, split_series, xy
from core.evaluation import score, write_summary
from core.models import fit_lgbm, fit_linear
from core.naive import naive_forecasts

OUT_DIR = OUTPUT_DIR / "test_evaluation"


def main() -> None:
    """Fit on train + valid, score every model and baseline once on test, save summary and predictions."""
    splits = split_series(load_features())
    fit_frame = pd.concat([splits["train"], splits["valid"]])  # contiguous, chronological
    test = splits["test"]
    X_fit, y_fit = xy(fit_frame, FINAL_FEATURES)
    X_test, y_test = xy(test, FINAL_FEATURES)
    fit_mean = float(y_fit.mean())

    kinds, preds = {}, {}
    for name, p in naive_forecasts(test, fit_mean).items():
        kinds[name], preds[name] = "baseline", p
    for model, fit in (("linear", fit_linear), ("lightgbm", fit_lgbm)):
        for loss in LOSSES:
            name = f"{model}_{loss}"
            kinds[name], preds[name] = "model", fit(X_fit, y_fit, X_test, loss)

    summary = {
        "split": "test",
        "features": FINAL_FEATURES,
        "fit_rows": len(fit_frame),
        "n": len(test),
        "fit_target_mean": fit_mean,
        "test_target_zero_share": float(np.mean(y_test == 0)),
        "results": {name: {"kind": kinds[name], **score(y_test, p)} for name, p in preds.items()},
    }
    write_summary(OUT_DIR / "summary.json", summary)
    pd.DataFrame({"unix_ts": test.index.to_numpy(), "actual": y_test, **preds}).to_parquet(
        OUT_DIR / "test_predictions.parquet", index=False
    )

    print(f"{'predictor':24s} {'kind':9s} {'MAE':>8s} {'RMSE':>8s} {'zero_pred':>10s}")
    for name, r in summary["results"].items():
        print(f"{name:24s} {r['kind']:9s} {r['mae']:8.4f} {r['rmse']:8.4f} {r['zero_prediction_share']:10.3f}")


if __name__ == "__main__":
    main()
