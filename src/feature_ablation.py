"""
Stage 4 feature-group ablation (1-minute resolution), in two sequential steps.

Every feature set is fitted with LightGBM (L2 and L1) and scored on the
validation split with MAE, RMSE and the share of exactly-zero predictions.

  Step 1  calendar ablation: every non-empty subset of (hour, dow_local,
          weekend), fitted alone and on top of all lags.
  Step 2  lag ablation: lag subsets with the calendar fixed to `hour` (step 1
          showed the other calendar features to be redundant), plus all lags
          without any calendar as reference. Calendar-then-lags is
          coordinate-wise, not a full grid: it assumes little interaction.

Everything but the feature set is held fixed: same train rows (lag warm-up
rows excluded for every set), split, hyperparameters and seed as Stage 3.
Linear regression is not part of this ablation.

Run: uv run python src/feature_ablation.py
Output: output/feature_ablation/summary.json
"""

import sys

import pandas as pd

from core.config import HOUR_SCALE, IMMEDIATE, LAG_COLS, LOSSES, OUTPUT_DIR, SEASONAL
from core.data import load_features, split_series, xy
from core.evaluation import score, write_summary
from core.models import fit_lgbm

OUT_JSON = OUTPUT_DIR / "feature_ablation" / "summary.json"

BOLD, RESET = ("\033[1;36m", "\033[0m") if sys.stdout.isatty() else ("", "")

CALENDAR_SETS = {
    "all_calendar": ["hour", "dow_local", "weekend"],
    "hour": ["hour"],
    "dow": ["dow_local"],
    "weekend": ["weekend"],
    "hour_dow": ["hour", "dow_local"],
    "hour_weekend": ["hour", "weekend"],
    "dow_weekend": ["dow_local", "weekend"],
}

LAG_SETS = {
    "all": LAG_COLS,
    "no_seasonal": IMMEDIATE + HOUR_SCALE,
    "immediate": IMMEDIATE,
    "sparse_hour": ["lag_1", "lag_15", *HOUR_SCALE],
    "seasonal": SEASONAL,
    "persistence": ["lag_1"],
}

# Result key -> (table title, {set name: feature columns}).
ABLATIONS = {
    "calendar_only": ("STEP 1a: CALENDAR ONLY", CALENDAR_SETS),
    "calendar_plus_all_lags": (
        "STEP 1b: CALENDAR + ALL LAGS",
        {f"{n} + all_lags": c + LAG_COLS for n, c in CALENDAR_SETS.items()},
    ),
    "lags_with_hour": (
        "STEP 2: LAG SETS (calendar = hour)",
        {
            **{n: ["hour"] + c for n, c in LAG_SETS.items()},
            "all_lags_without_hour": LAG_COLS,  # reference
        },
    ),
}


def run(splits: dict[str, pd.DataFrame], title: str, sets: dict[str, list[str]]) -> dict:
    """Fit every set under both losses on train, score on valid, print one table per loss."""
    train, valid = splits["train"], splits["valid"]
    results = {}
    for loss in LOSSES:
        model_name = f"lightgbm_{loss}"
        print(f"\n{BOLD}==== {title} | {model_name} ===={RESET}")
        print(f"{'feature set':28s} {'n_feat':>6s} {'MAE':>8s} {'RMSE':>8s} {'zero_pred':>10s}")
        results[model_name] = {}
        for name, columns in sets.items():
            X_train, y_train = xy(train, columns)
            X_valid, y_valid = xy(valid, columns)
            res = {"n_features": len(columns), **score(y_valid, fit_lgbm(X_train, y_train, X_valid, loss))}
            results[model_name][name] = res
            print(
                f"{name:28s} {res['n_features']:6d} {res['mae']:8.4f} {res['rmse']:8.4f} "
                f"{res['zero_prediction_share']:10.3f}"
            )
    return results


def main() -> None:
    """Run all ablation steps in order and save the reproducible summary."""
    splits = split_series(load_features())
    summary = {"split": "validation"}
    for key, (title, sets) in ABLATIONS.items():
        summary[key] = run(splits, title, sets)
    write_summary(OUT_JSON, summary)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
