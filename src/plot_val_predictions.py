"""Plot actual and predicted validation values for Stage 2 and Stage 3."""

from math import ceil
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PREDICTIONS = REPO_ROOT / "output" / "baselines" / "validation_predictions.parquet"
MODEL_PREDICTIONS = REPO_ROOT / "output" / "diagnostic_models" / "validation_predictions.parquet"
FIG_DIR = REPO_ROOT / "output" / "validation_predictions" / "figs"

# Week numbers are counted from the start of the 13-week validation period.
VALIDATION_WEEK = 1  # 1 through 13
DAY_OF_WEEK = None  # None = whole week; 1 = Monday through 7 = Sunday
PREDICTIONS_TO_PLOT = None  # None = all methods; otherwise use column names below
LOCAL_TIMEZONE = "America/Vancouver"

METHOD_LABELS = {
    "always_zero": "Always zero",
    "always_mean": "Training mean",
    "persistence": "Persistence (1 minute)",
    "seasonal_naive_daily": "Daily seasonal naive",
    "seasonal_naive_weekly": "Weekly seasonal naive",
    "linear_regression": "Linear regression",
    "lightgbm": "LightGBM",
}
DAY_NAMES = {day: name for day, name in enumerate(
    ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    start=1,
)}


def load_validation_predictions() -> pd.DataFrame:
    """Load and align the baseline and diagnostic-model prediction artifacts."""
    baselines = pd.read_parquet(BASELINE_PREDICTIONS)
    models = pd.read_parquet(MODEL_PREDICTIONS)
    required = {"unix_ts", "actual"}
    for label, frame in (("baseline", baselines), ("model", models)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} prediction artifact is missing columns: {sorted(missing)}")
        if frame["unix_ts"].duplicated().any():
            raise ValueError(f"{label} prediction artifact contains duplicate timestamps")
        if frame["unix_ts"].is_monotonic_increasing is False:
            raise ValueError(f"{label} prediction timestamps are not chronological")

    if not baselines["unix_ts"].equals(models["unix_ts"]):
        raise ValueError("Baseline and model prediction artifacts use different timestamps")
    if not np.array_equal(baselines["actual"].to_numpy(), models["actual"].to_numpy()):
        raise ValueError("Baseline and model prediction artifacts have different actual values")

    duplicate_methods = (set(baselines.columns) & set(models.columns)) - required
    if duplicate_methods:
        raise ValueError(f"Prediction method names appear in both artifacts: {sorted(duplicate_methods)}")
    return pd.concat(
        [
            baselines.reset_index(drop=True),
            models.drop(columns=["unix_ts", "actual"]).reset_index(drop=True),
        ],
        axis=1,
    )


def select_validation_window(
    predictions: pd.DataFrame,
    week_number: int,
    day_of_week: int | None,
    timezone: str = LOCAL_TIMEZONE,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Select a local-calendar validation week and optionally one weekday."""
    if not 1 <= week_number <= 13:
        raise ValueError("week_number must be between 1 and 13")
    if day_of_week is not None and day_of_week not in DAY_NAMES:
        raise ValueError("day_of_week must be None or an integer from 1 (Monday) to 7 (Sunday)")

    utc_times = pd.to_datetime(predictions["unix_ts"], unit="s", utc=True)
    local_times = pd.DatetimeIndex(utc_times).tz_convert(timezone)
    local_dates = local_times.tz_localize(None).normalize()
    elapsed_days = np.asarray((local_dates - local_dates[0]).days)
    week_numbers = elapsed_days // 7 + 1
    weekdays = local_times.dayofweek + 1
    mask = week_numbers == week_number
    if day_of_week is not None:
        mask &= weekdays == day_of_week
    selected = predictions.loc[mask].reset_index(drop=True)
    selected_times = local_times[mask]
    if selected.empty:
        raise ValueError("The requested week/day has no validation observations")
    return selected, selected_times


def plot_predictions(
    predictions: pd.DataFrame,
    local_times: pd.DatetimeIndex,
    week_number: int,
    day_of_week: int | None,
    methods: list[str] | None = None,
) -> Path:
    """Save small-multiple plots of actual values and selected predictions."""
    available = [column for column in predictions.columns if column in METHOD_LABELS]
    selected_methods = available if methods is None else methods
    unknown = set(selected_methods) - set(available)
    if unknown:
        raise ValueError(f"Requested prediction methods are unavailable: {sorted(unknown)}")
    if not selected_methods:
        raise ValueError("At least one prediction method must be selected")

    ncols = 2
    nrows = ceil(len(selected_methods) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 3.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    timezone = ZoneInfo(LOCAL_TIMEZONE)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9, tz=timezone)
    formatter = mdates.DateFormatter("%a %d %H:%M %Z", tz=timezone)

    for ax, method in zip(axes.flat, selected_methods):
        ax.plot(local_times, predictions["actual"], color="black", linewidth=0.8, label="Actual")
        ax.plot(
            local_times,
            predictions[method],
            linewidth=0.75,
            label="Prediction",
        )
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel("avg_rate (L/min)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=False)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    for ax in axes.flat[len(selected_methods):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(f"Local time ({LOCAL_TIMEZONE})")
    fig.autofmt_xdate(rotation=25, ha="right")

    period = f"validation week {week_number}"
    filename = f"week_{week_number:02d}"
    if day_of_week is not None:
        period += f", {DAY_NAMES[day_of_week]}"
        filename += f"_day_{day_of_week}_{DAY_NAMES[day_of_week].lower()}"
    fig.suptitle(f"Actual and predicted water consumption — {period}")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / f"{filename}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Plot the configured validation week/day for each available method."""
    predictions = load_validation_predictions()
    selected, local_times = select_validation_window(
        predictions,
        week_number=VALIDATION_WEEK,
        day_of_week=DAY_OF_WEEK,
    )
    output_path = plot_predictions(
        selected,
        local_times,
        week_number=VALIDATION_WEEK,
        day_of_week=DAY_OF_WEEK,
        methods=PREDICTIONS_TO_PLOT,
    )
    print(f"Validation prediction plot saved to {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
