"""Pairwise Spearman correlation between model features (training split only).

First, model-free proxy for feature redundancy. Pairwise and monotonic only:
cyclic encodings (hour vs. hour_sin/hour_cos) are not monotonically related,
so low values there do not mean the features carry different information.

Run: uv run python src/feature_correlation.py
Output: output/feature_correlation/spearman.csv and spearman.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from objective_ablation import FEATURES_LINEAR, FEATURES_TREE
from splits import load_features, split_series

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "feature_correlation"

# Union of both models' feature lists, calendar before lags.
FEATURES = list(dict.fromkeys(FEATURES_TREE + FEATURES_LINEAR))
FEATURES.sort(key=lambda c: c.startswith("lag_"))


def spearman_matrix(train: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation matrix of the model features (ties get average ranks)."""
    return train[FEATURES].astype(float).corr(method="spearman")


def plot_matrix(corr: pd.DataFrame, path: Path) -> None:
    """Annotated heatmap of a correlation matrix."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.75 * n + 2, 0.75 * n + 1))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("Feature Spearman correlation (training split)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Compute on the training split and save matrix and heatmap."""
    corr = spearman_matrix(split_series(load_features())["train"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUT_DIR / "spearman.csv")
    plot_matrix(corr, OUT_DIR / "spearman.pdf")
    print(f"Saved {OUT_DIR.relative_to(REPO_ROOT)}/spearman.csv and spearman.pdf")


if __name__ == "__main__":
    main()
