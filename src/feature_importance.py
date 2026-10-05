"""
Stage 5 feature importance: grouped TreeSHAP for LightGBM, L2 and L1.

Each loss is fitted with the Stage 3 settings on train and analysed on the
validation split (the test split is untouched), for two feature sets: all
features, to cross-check the Stage 4 ablation, and the final set.

TreeSHAP comes from LightGBM's `predict(pred_contrib=True)` and is reported per
feature group: the mean |row-wise summed contribution| of the group's features
(single-lag values are unstable because the lags are correlated), and the
group's share of the total.

Run: uv run python src/feature_importance.py
Output: output/feature_importance/summary.json
"""

import numpy as np

from core.config import (
    FEATURES_TREE,
    FINAL_FEATURES,
    HOUR_SCALE,
    IMMEDIATE,
    LOSSES,
    OUTPUT_DIR,
    SEASONAL,
)
from core.data import load_features, split_series, xy
from core.evaluation import score, write_summary
from core.models import train_lgbm

OUT_JSON = OUTPUT_DIR / "feature_importance" / "summary.json"

# Feature set -> (model columns, {group: member columns}); groups partition the columns.
FEATURE_SETS = {
    "all_features": (
        FEATURES_TREE,
        {
            "hour": ["hour"],
            "day_of_week": ["dow_local", "weekend"],
            "immediate_lags": IMMEDIATE,
            "hour_scale_lags": HOUR_SCALE,
            "seasonal_lags": SEASONAL,
        },
    ),
    "final": (FINAL_FEATURES, {"hour": ["hour"], "immediate_lags": IMMEDIATE}),
}


def shap_groups(model, X, columns, groups) -> dict:
    """Mean |TreeSHAP| per group (row-wise summed contributions) and its share of the total."""
    contrib = model.predict(X, pred_contrib=True)  # last column is the expected value
    assert np.allclose(contrib.sum(axis=1), model.predict(X, raw_score=True)), "SHAP not additive"
    mean_abs = {
        name: np.abs(contrib[:, [columns.index(c) for c in members]].sum(axis=1)).mean()
        for name, members in groups.items()
    }
    total = sum(mean_abs.values())
    return {name: {"mean_abs_shap": v, "share": v / total} for name, v in mean_abs.items()}


def print_groups(title: str, groups: dict) -> None:
    """One table per model, sorted by mean |SHAP|."""
    print(f"\n{title}")
    print(f"{'group':18s} {'mean|SHAP|':>11s} {'share':>7s}")
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["mean_abs_shap"]):
        print(f"{name:18s} {g['mean_abs_shap']:11.4f} {g['share']:7.3f}")


def main() -> None:
    """Fit each (loss, feature set), compute group SHAP importances on validation, save the summary."""
    splits = split_series(load_features())
    summary = {"split": "validation", "models": {}}
    for loss in LOSSES:
        name = f"lightgbm_{loss}"
        summary["models"][name] = {}
        for set_name, (columns, groups) in FEATURE_SETS.items():
            X_train, y_train = xy(splits["train"], columns)
            X_valid, y_valid = xy(splits["valid"], columns)
            model = train_lgbm(X_train, y_train, loss)
            result = {
                "valid": score(y_valid, model.predict(X_valid)),
                "groups": shap_groups(model, X_valid, columns, groups),
            }
            summary["models"][name][set_name] = result
            print_groups(f"{name} | {set_name}", result["groups"])

    write_summary(OUT_JSON, summary)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
