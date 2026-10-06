"""
Residual ACF of the final-feature LightGBM models (L2 and L1), fitted on train,
next to the ACF of the target, both on the validation split.

Output: output/residual_acf/summary.json and figs/residual_acf.png (lags 1-60)
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from core.config import FINAL_FEATURES, LAGS, LOSSES, OUTPUT_DIR
from core.data import load_features, split_series, xy
from core.evaluation import write_summary
from core.models import fit_lgbm

OUT_DIR = OUTPUT_DIR / "residual_acf"
PLOT_MAX_LAG = 60


def acf_curve(x: np.ndarray) -> np.ndarray:
    """Sample ACF of x for lags 0..max(LAGS)."""
    return acf(x, nlags=max(LAGS), fft=True)


def at_lags(curve: np.ndarray) -> dict[str, float]:
    """Pick the candidate lags out of an ACF curve."""
    return {str(k): float(curve[k]) for k in LAGS}


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit both losses on train; return the summary and the full ACF curves (target first)."""
    X_train, y_train = xy(splits["train"], FINAL_FEATURES)
    X_valid, y_valid = xy(splits["valid"], FINAL_FEATURES)
    curves = {"target": acf_curve(y_valid)}
    for loss in LOSSES:
        residuals = y_valid - fit_lgbm(X_train, y_train, X_valid, loss)
        curves[f"lightgbm_{loss}"] = acf_curve(residuals)

    summary = {
        "split": "validation",
        "features": "final",
        "n": len(y_valid),
        "white_noise_band": 1.96 / np.sqrt(len(y_valid)),
        "target_acf": at_lags(curves["target"]),
        "residual_acf": {name: at_lags(c) for name, c in curves.items() if name != "target"},
    }
    return summary, curves


def plot_correlogram(curves: dict[str, np.ndarray], band: float) -> None:
    """Left: target and residual ACFs; right: residuals only, zoomed. Dashed: white-noise band."""
    lags = np.arange(1, PLOT_MAX_LAG + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {name: f"C{i}" for i, name in enumerate(curves)}  # same colour per series in both panels
    for ax, names in zip(axes, [list(curves), [n for n in curves if n != "target"]]):
        for name in names:
            ax.plot(
                lags, curves[name][1 : PLOT_MAX_LAG + 1], marker="o", markersize=3, linewidth=0.8,
                color=colors[name], label=name,
            )
        ax.axhline(0, color="black", linewidth=0.5)
        for sign in (1, -1):
            ax.axhline(sign * band, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Lag (minutes)")
        ax.set_ylabel("ACF")
        ax.legend()
    axes[0].set_title("Target and residuals")
    axes[1].set_title("Residuals only (zoom)")
    fig.suptitle("Validation ACF, final feature set; dashed: white-noise band +-1.96/sqrt(n)")
    fig.tight_layout()
    (OUT_DIR / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "figs" / "residual_acf.png", dpi=150)
    plt.close(fig)


def print_table(summary: dict) -> None:
    """ACF of the target and of each model's residuals at the candidate lags."""
    names = list(summary["residual_acf"])
    print(f"n={summary['n']}  white-noise band +-{summary['white_noise_band']:.4f}")
    print(f"{'lag':>6s} {'target':>8s} " + " ".join(f"{n:>14s}" for n in names))
    for k in LAGS:
        row = " ".join(f"{summary['residual_acf'][n][str(k)]:14.4f}" for n in names)
        print(f"{k:6d} {summary['target_acf'][str(k)]:8.4f} {row}")


def main() -> None:
    summary, curves = run(split_series(load_features()))
    write_summary(OUT_DIR / "summary.json", summary)
    plot_correlogram(curves, summary["white_noise_band"])
    print_table(summary)


if __name__ == "__main__":
    main()
