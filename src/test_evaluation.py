"""
Single evaluation on the test split: the final models (LightGBM and linear
regression, L2 and L1, on hour + lags 1-15) are fitted on train + valid and
scored next to the five baselines; always-mean uses the train + valid mean.

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


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Fit the final models on train + valid; score them and the baselines on test."""
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
    pred_frame = pd.DataFrame({"unix_ts": test.index.to_numpy(), "actual": y_test, **preds})
    return summary, pred_frame


def main() -> None:
    summary, predictions = run(split_series(load_features()))
    write_summary(OUT_DIR / "summary.json", summary)
    predictions.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
    print(f"{'predictor':24s} {'kind':9s} {'MAE':>8s} {'RMSE':>8s} {'zero_pred':>10s}")
    for name, r in summary["results"].items():
        print(f"{name:24s} {r['kind']:9s} {r['mae']:8.4f} {r['rmse']:8.4f} {r['zero_prediction_share']:10.3f}")


if __name__ == "__main__":
    main()
