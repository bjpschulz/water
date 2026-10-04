"""
Stage 5 feature importance (1-minute resolution): LightGBM, L2 and L1.

Each loss is fitted with the Stage 3 settings on train and analysed on the
validation split (the test split is untouched), for two feature sets: all
features (calendar + full lag set), to cross-check the Stage 4 ablation -- do
the features it found redundant also rate low here? -- and the final set.

Importance is TreeSHAP via LightGBM's `predict(pred_contrib=True)`, reported per
feature group: mean |contribution| of the group, from the row-wise summed
contributions of its features (the lags are strongly correlated, so single-lag
values are unstable), and the group's share of the total.

Run: uv run python src/feature_importance.py
Output: output/feature_importance/summary.json (groups are defined by FEATURE_SETS)
"""

from pathlib import Path

import numpy as np

from diagnostic_models import FEATURES_TREE, LGBM_OBJECTIVES, _xy, score, train_lgbm
from feature_ablation import HOUR_SCALE, IMMEDIATE, SEASONAL
from splits import load_features, split_series
from ts_utils import write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "output" / "feature_importance" / "summary.json"

# Feature set -> (model columns, {group: member columns}).
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
    "final": (["hour"] + IMMEDIATE, {"hour": ["hour"], "immediate_lags": IMMEDIATE}),
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
    for suffix, objective in LGBM_OBJECTIVES.items():
        name = f"lightgbm_{suffix}"
        summary["models"][name] = {}
        for set_name, (columns, groups) in FEATURE_SETS.items():
            X_train, y_train = _xy(splits["train"], columns)
            X_valid, y_valid = _xy(splits["valid"], columns)
            model = train_lgbm(X_train, y_train, objective)
            result = {
                "valid": score(y_valid, model.predict(X_valid)),
                "groups": shap_groups(model, X_valid, columns, groups),
            }
            summary["models"][name][set_name] = result
            print_groups(f"{name} | {set_name}", result["groups"])

    write_summary(OUT_JSON, summary)
    print(f"\nSaved {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
