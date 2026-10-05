"""
Stage 5 residual ACF: how much temporal structure do the final LightGBM
models leave unexplained?

Both losses (L2, L1) are fitted on the final feature set with the Stage 3
settings on train; residuals y - y_hat are taken on the validation split (the
test split is untouched) and their ACF is computed at the candidate lags
(`config.LAGS`), next to the ACF of the target itself on the same window. The
validation slice is a contiguous 1-minute grid, so lags are exact.

The +-1.96/sqrt(n) band is the white-noise reference only: with this many rows
and event-structured data it is tiny, so values inside it mean "no structure",
values outside it are not automatically practically relevant. Residual
autocorrelation is a descriptive diagnostic, not a significance test.

Run: uv run python src/residual_acf.py
Output: output/residual_acf/summary.json, figs/residual_acf.png (correlogram, lags 1-60)
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf

from core.config import FINAL_FEATURES, LAGS, LOSSES, OUTPUT_DIR
from core.data import load_features, split_series, xy
from core.evaluation import write_summary
from core.models import train_lgbm

OUT_JSON = OUTPUT_DIR / "residual_acf" / "summary.json"
FIG_DIR = OUTPUT_DIR / "residual_acf" / "figs"
PLOT_MAX_LAG = 60


def acf_curve(x: np.ndarray) -> np.ndarray:
    """Sample ACF of x for lags 0..max(LAGS)."""
    return acf(x, nlags=max(LAGS), fft=True)


def at_lags(curve: np.ndarray) -> dict[str, float]:
    """Pick the candidate lags out of an ACF curve."""
    return {str(k): float(curve[k]) for k in LAGS}


def plot_correlogram(curves: dict[str, np.ndarray], band: float) -> None:
    """Left: target and residual ACFs on one scale; right: residuals only, zoomed. Dashed = white-noise band."""
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
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "residual_acf.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Fit both losses, compute residual and target ACF on validation, save the summary."""
    splits = split_series(load_features())
    X_train, y_train = xy(splits["train"], FINAL_FEATURES)
    X_valid, y_valid = xy(splits["valid"], FINAL_FEATURES)

    target_curve = acf_curve(y_valid)
    curves = {}
    for loss in LOSSES:
        model = train_lgbm(X_train, y_train, loss)
        curves[f"lightgbm_{loss}"] = acf_curve(y_valid - model.predict(X_valid))

    summary = {
        "split": "validation",
        "features": "final",
        "n": len(y_valid),
        "white_noise_band": 1.96 / np.sqrt(len(y_valid)),
        "target_acf": at_lags(target_curve),
        "residual_acf": {name: at_lags(c) for name, c in curves.items()},
    }
    plot_correlogram({"target": target_curve, **curves}, summary["white_noise_band"])
    write_summary(OUT_JSON, summary)
    names = list(summary["residual_acf"])
    print(f"n={summary['n']}  white-noise band +-{summary['white_noise_band']:.4f}")
    print(f"{'lag':>6s} {'target':>8s} " + " ".join(f"{n:>14s}" for n in names))
    for k in LAGS:
        row = " ".join(f"{summary['residual_acf'][n][str(k)]:14.4f}" for n in names)
        print(f"{k:6d} {summary['target_acf'][str(k)]:8.4f} {row}")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
