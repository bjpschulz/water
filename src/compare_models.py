"""
Diagnostic models: linear regression and LightGBM, each under an L2 and an L1
loss, on calendar + all lags, fitted on train and scored on validation.
Linear L1 is median regression; only the loss differs within a model class.

Output: output/compare_models/summary.json and validation_predictions.parquet
"""

import pandas as pd

from core.config import FEATURES_LINEAR, FEATURES_TREE, LOSSES, OUTPUT_DIR
from core.data import load_features, split_series, xy
from core.evaluation import score, write_summary
from core.models import fit_lgbm, fit_linear

OUT_DIR = OUTPUT_DIR / "compare_models"


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Fit every (model, loss) pair on train and score it on validation."""
    train, valid = splits["train"], splits["valid"]
    preds = {}
    for model, features, fit in (("linear", FEATURES_LINEAR, fit_linear), ("lightgbm", FEATURES_TREE, fit_lgbm)):
        X_train, y_train = xy(train, features)
        X_valid, y_valid = xy(valid, features)
        for loss in LOSSES:
            preds[f"{model}_{loss}"] = fit(X_train, y_train, X_valid, loss)

    summary = {
        "split": "validation",
        "models": {name: {"loss": name.rsplit("_", 1)[1], **score(y_valid, p)} for name, p in preds.items()},
    }
    pred_frame = pd.DataFrame({"unix_ts": valid.index.to_numpy(), "actual": y_valid, **preds})
    return summary, pred_frame


def main() -> None:
    summary, predictions = run(split_series(load_features()))
    write_summary(OUT_DIR / "summary.json", summary)
    predictions.to_parquet(OUT_DIR / "validation_predictions.parquet", index=False)
    for name, m in summary["models"].items():
        print(f"{name:12s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} zero_pred={m['zero_prediction_share']:.3f}")


if __name__ == "__main__":
    main()
