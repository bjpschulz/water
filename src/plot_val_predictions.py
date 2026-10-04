"""Plot actual vs. predicted water consumption for one validation window.

Usage: uv run src/plot_val_predictions.py --week 2 --day 2 --hours 16-22 --methods persistence linear_l2
"""

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PREDICTIONS = REPO_ROOT / "output" / "baselines" / "validation_predictions.parquet"
MODEL_PREDICTIONS = REPO_ROOT / "output" / "diagnostic_models" / "validation_predictions.parquet"
FIG_DIR = REPO_ROOT / "output" / "validation_predictions" / "figs"

LOCAL_TIMEZONE = "America/Vancouver"
N_VALIDATION_WEEKS = 13
DEFAULT_METHODS = [
    "always_zero",
    "persistence",
    "linear_l1",
    "linear_l2",
    "lightgbm_l1",
    "lightgbm_l2",
]

METHOD_LABELS = {
    "always_zero": "Always zero",
    "always_mean": "Training mean",
    "persistence": "Persistence (1 minute)",
    "seasonal_naive_daily": "Daily seasonal naive",
    "seasonal_naive_weekly": "Weekly seasonal naive",
    "linear_l2": "Linear regression (L2)",
    "linear_l1": "Linear regression (L1)",
    "lightgbm_l2": "LightGBM (L2)",
    "lightgbm_l1": "LightGBM (L1)",
}
DAY_NAMES = dict(
    enumerate(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"), start=1)
)


@dataclass(frozen=True)
class Window:
    """Local-calendar slice of the validation period.

    `week` counts from the start of validation (1-13); `day` is 1 (Monday) to 7 (Sunday)
    or None for the whole week; `hours` is a local [start, end) hour range or None.
    """

    week: int
    day: int | None = None
    hours: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.week <= N_VALIDATION_WEEKS:
            raise ValueError(f"week must be between 1 and {N_VALIDATION_WEEKS}")
        if self.day is not None and self.day not in DAY_NAMES:
            raise ValueError("day must be None or an integer from 1 (Monday) to 7 (Sunday)")
        if self.hours is not None and not 0 <= self.hours[0] < self.hours[1] <= 24:
            raise ValueError("hours must satisfy 0 <= start < end <= 24")

    def mask(self, local_time: pd.Series) -> pd.Series:
        """Boolean mask of rows inside the window (local_time is tz-aware)."""
        dates = local_time.dt.tz_localize(None).dt.normalize()
        mask = (dates - dates.iloc[0]).dt.days // 7 + 1 == self.week
        if self.day is not None:
            mask &= local_time.dt.dayofweek + 1 == self.day
        if self.hours is not None:
            mask &= (local_time.dt.hour >= self.hours[0]) & (local_time.dt.hour < self.hours[1])
        return mask

    def title(self) -> str:
        text = f"validation week {self.week}"
        if self.day is not None:
            text += f", {DAY_NAMES[self.day]}"
        if self.hours is not None:
            text += f", {self.hours[0]:02d}:00–{self.hours[1]:02d}:00 local time"
        return text

    def slug(self) -> str:
        text = f"week_{self.week:02d}"
        if self.day is not None:
            text += f"_day_{self.day}_{DAY_NAMES[self.day].lower()}"
        if self.hours is not None:
            text += f"_hours_{self.hours[0]:02d}-{self.hours[1]:02d}"
        return text


def load_validation_predictions() -> pd.DataFrame:
    """Load baseline and model predictions, aligned, with a tz-aware `local_time` column."""
    baselines = pd.read_parquet(BASELINE_PREDICTIONS)
    models = pd.read_parquet(MODEL_PREDICTIONS)
    if not baselines["unix_ts"].equals(models["unix_ts"]):
        raise ValueError("Baseline and model prediction artifacts use different timestamps")
    predictions = pd.concat(
        [
            baselines.reset_index(drop=True),
            models.drop(columns=["unix_ts", "actual"]).reset_index(drop=True),
        ],
        axis=1,
    )
    utc = pd.to_datetime(predictions["unix_ts"], unit="s", utc=True)
    predictions["local_time"] = utc.dt.tz_convert(LOCAL_TIMEZONE)
    return predictions


def plot_predictions(predictions: pd.DataFrame, window: Window, methods: list[str]) -> Figure:
    """Small multiples (two columns) of actual vs. predicted values inside `window`."""
    unknown = set(methods) - (set(predictions.columns) & set(METHOD_LABELS))
    if unknown:
        raise ValueError(f"Requested prediction methods are unavailable: {sorted(unknown)}")
    if not methods:
        raise ValueError("At least one prediction method must be selected")

    data = predictions.loc[window.mask(predictions["local_time"])]
    if data.empty:
        raise ValueError("The requested window has no validation observations")

    ncols = 2
    nrows = ceil(len(methods) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15, 3.2 * nrows), sharex=True, sharey=True, squeeze=False
    )
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9, tz=LOCAL_TIMEZONE)
    formatter = mdates.DateFormatter("%a %d %H:%M %Z", tz=LOCAL_TIMEZONE)

    for ax, method in zip(axes.flat, methods):
        ax.plot(data["local_time"], data["actual"], color="tab:blue", linewidth=0.8, label="Actual")
        ax.plot(data["local_time"], data[method], color="tab:orange", linewidth=0.75, label="Prediction")
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel("avg_rate (L/min)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=False)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(f"Local time ({LOCAL_TIMEZONE})")
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.suptitle(f"Actual and predicted water consumption — {window.title()}")
    fig.tight_layout()
    return fig


def parse_hours(text: str) -> tuple[int, int]:
    """Parse 'START-END' (e.g. '16-22') into an hour pair."""
    start, end = text.split("-")
    return int(start), int(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--week", type=int, default=2, help="validation week, 1-13")
    parser.add_argument("--day", type=int, help="1 (Monday) to 7 (Sunday); default whole week")
    parser.add_argument("--hours", type=parse_hours, help="local hour range START-END, e.g. 16-22")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="prediction columns")
    args = parser.parse_args()

    window = Window(args.week, args.day, args.hours)
    fig = plot_predictions(load_validation_predictions(), window, args.methods)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / f"{window.slug()}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Validation prediction plot saved to {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
