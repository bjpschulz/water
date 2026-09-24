"""
Reproducible EDA for the AMPds2 whole-house water (WHW) data, at native
1-minute resolution.

Purpose relative to the project's research questions (see AGENTS.md): this
script quantifies (a) data integrity at the raw 1-minute grid, (b) the
effect of the 2012 meter replacement, (c) the extent of zero-inflation and
event structure in whole-house water consumption, and (d) autocorrelation
at 1-minute lags -- including a split between the "raw" ACF and the ACF of
the (avg_rate > 0) occurrence indicator, to help separate genuine short-run
persistence from the trivial "zero tends to follow zero" effect.

The modelling resolution is not yet settled: native 1-minute is the
working default, with the decision gated on the Stage 2-4 baseline /
diagnostic comparison (see STATE.md, "Open decision: forecasting
horizon"). This script therefore measures the 1-minute structure -- an
input to that decision, not a final modelling choice.

Timezone convention (settled fact, AGENTS.md): unix_ts is true Unix/UTC
time and the household is in America/Vancouver. The master grid, lags,
and the stored artifact are keyed by unix_ts (the UTC grid); calendar
grouping (temporal profiles, calendar figures) and the stored
datetime_local column use local time. A UTC datetime is derivable from
unix_ts in one line and is not stored.

Run: uv run python src/eda_whw.py

Outputs:
  data/processed/whw_v100.parquet   cleaned analysis period (1-min grid;
                                    index unix_ts, columns datetime_local,
                                    counter, avg_rate)
  data/processed/whw_v100.csv       identical content as CSV, for inspection
  results/eda_whw/summary.json      all key numbers
  results/eda_whw/figs/*.png        figures

The analysis period starts at V100_CLEAN_UNIX_TS, not at the meter-swap
timestamp: the first V100 minutes contain one sub-pulse counter-settling
artifact, after which all values sit exactly on the 0.5 L pulse grid. See
the "clean_start" block in summary.json's meter_transition section.
"""

#%%
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window when run as a script

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from ts_utils import acf_fft, distribution_summary, segment_runs, to_serializable

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Water_WHW.csv"
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "whw_v100.parquet"
CSV_PATH = REPO_ROOT / "data" / "processed" / "whw_v100.csv"  # twin of the parquet, for eyeballing
OUT_DIR = REPO_ROOT / "results" / "eda_whw"
FIG_DIR = OUT_DIR / "figs"

V100_UNIX_TS = 1_342_287_780  # 2012-07-14: documented switch to the V100 water meter

# First timestamp at which the V100 series is fully settled on the 0.5 L
# pulse grid: immediately after the swap the counter shows one single
# sub-pulse increment (< 0.5 L, at unix_ts 1342310520, avg_rate 0.053) as
# it settles onto the new pulse grid.
V100_CLEAN_UNIX_TS = 1_342_310_580

LOCAL_TZ = "America/Vancouver"

# Candidate lags for THIS (1-minute-resolution) analysis only. Chosen to
# span short-run persistence (1-30 min) up to daily/weekly cycles.
# NOTE: if the project moves to 15-minute aggregated blocks, this set is to be
# superseded by a block-resolution lag set (see AGENTS.md) -- it is not
# reused as-is at the coarser resolution.
CANDIDATE_LAGS = [1, 5, 15, 30, 60, 1440, 10080]
ACF_MAX_LAG = 10080  # one week, in minutes
SECONDS_PER_MINUTE = 60
PULSE_SIZE_L = 0.5  # V100 meter pulse size, in liters

#%%

def load_water() -> pd.DataFrame:
    """Load the raw water CSV and attach tz-aware UTC and local datetime columns, sorted by time."""
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["unix_ts"], unit="s", utc=True)
    df["datetime_local"] = df["datetime"].dt.tz_convert(LOCAL_TZ)
    return df.sort_values("unix_ts").reset_index(drop=True)

def check_raw(df: pd.DataFrame) -> dict:
    """
    Basic integrity checks on the full (both meter periods) raw series:
    grid regularity, duplicate/missing timestamps, missing values, sign
    checks, and consistency between the cumulative counter and avg_rate.
    """
    diffs = df["unix_ts"].diff().to_numpy()[1:]
    gaps = diffs[diffs != SECONDS_PER_MINUTE]
    largest_gap_idx = int(np.argmax(gaps)) if len(gaps) else None
    out = {
        "n_rows": len(df),
        "columns": df.columns.tolist(),
        "missing_values": df[["unix_ts", "counter", "avg_rate", "inst_rate"]]
        .isna()
        .sum()
        .to_dict(),
        "unix_ts_monotone_increasing": bool(df["unix_ts"].is_monotonic_increasing),
        "duplicate_unix_ts": int(df["unix_ts"].duplicated().sum()),
        "n_rows_not_on_1min_grid": int(len(gaps)),
        "missing_minutes_total": int(
            (diffs[diffs > SECONDS_PER_MINUTE] / SECONDS_PER_MINUTE - 1).sum()
        ),
        "start": df["datetime"].iloc[0],
        "end": df["datetime"].iloc[-1],
        "expected_rows_730_days": 730 * 24 * 60,
    }
    if largest_gap_idx is not None:
        gap_row = df.iloc[np.flatnonzero(diffs != SECONDS_PER_MINUTE)[largest_gap_idx] + 1]
        out["largest_gap"] = {
            "at": gap_row["datetime"],
            "seconds_since_previous": int(gaps[largest_gap_idx]),
        }
    out["negative_avg_rate"] = int((df["avg_rate"] < 0).sum())
    out["negative_counter_diffs"] = int((df["counter"].diff() < 0).sum())

    # Cross-check: counter differences should match avg_rate almost exactly
    counter_vs_rate = (df["counter"].diff() - df["avg_rate"]).abs()
    out["max_abs_counter_diff_minus_avg_rate"] = float(counter_vs_rate.iloc[1:].max())
    return out


def analyze_transition(df: pd.DataFrame) -> dict:
    """
    Characterize the 2012-07-14 meter replacement: compare pulse sizes
    (smallest and most common positive counter increments) before and
    after, and check that the cumulative counter itself stayed continuous
    across the swap (no reset to zero, no jump).
    """
    old = df[df["unix_ts"] < V100_UNIX_TS]
    v100 = df[df["unix_ts"] >= V100_UNIX_TS]

    def pulse_summary(period: pd.DataFrame) -> dict:
        """Empirically estimate the meter's pulse size from positive counter increments."""
        inc = period["counter"].diff().dropna()
        pos = inc[inc > 0]
        rounded = pos.round(3)
        top = rounded.value_counts().head(3)
        return {
            "n_rows": len(period),
            "smallest_positive_counter_increment_L": float(pos.min()) if len(pos) else None,
            "most_common_positive_increments_L": {
                float(k): int(v) for k, v in top.items()
            },
        }

    transition_row = df.index[df["unix_ts"] >= V100_UNIX_TS][0]
    counter_before = df["counter"].iloc[transition_row - 1]
    counter_after = df["counter"].iloc[transition_row]

    # Immediately after the swap, the counter shows one single increment
    # smaller than the expected 0.5 L V100 pulse size, as it settles onto
    # the new pulse grid.
    v100_increments = v100["counter"].diff()
    sub_pulse = v100_increments[(v100_increments > 0) & (v100_increments < 0.5)]
    sub_pulse_note = {
        "count": int(len(sub_pulse)),
        "timestamps": v100.loc[sub_pulse.index, "datetime"].tolist(),
        "unix_timestamps": v100.loc[sub_pulse.index, "unix_ts"].tolist(),
        "note": (
            "One-off counter settling onto the 0.5 L pulse grid shortly after the "
            "meter swap; all other V100 increments are multiples of 0.5 L."
        ),
    }

    # Everything from V100_CLEAN_UNIX_TS onward must sit exactly on the
    # 0.5 L pulse grid: verify that claim rather than assume it.
    settled = df[df["unix_ts"] >= V100_CLEAN_UNIX_TS]
    settled_increments = settled["counter"].diff()
    off_grid_counter = int(
        (settled_increments.dropna() % PULSE_SIZE_L > 1e-9).sum()
    )
    off_grid_avg_rate = int((settled["avg_rate"] % PULSE_SIZE_L > 1e-9).sum())

    return {
        "v100_unix_ts": V100_UNIX_TS,
        "v100_datetime": df["datetime"].iloc[transition_row],
        "clean_start_unix_ts": V100_CLEAN_UNIX_TS,
        "rows_excluded_before_clean_start": int(len(v100[v100["unix_ts"] < V100_CLEAN_UNIX_TS])),
        "rows_excluded_before_transition": int(len(old)),
        "rows_kept_from_transition_onward": int(len(v100)),
        "clean_start": {
            "note": (
                "Analysis period and stored artifact start at V100_CLEAN_UNIX_TS, "
                "the first timestamp after which all values sit on the 0.5 L pulse "
                "grid; the rows between the meter swap and this point are used only "
                "to document the transition."
            ),
            "off_grid_counter_increments": off_grid_counter,
            "off_grid_avg_rate_values": off_grid_avg_rate,
        },
        "rows_excluded_before_transition": int(len(old)),
        "rows_kept_from_transition_onward": int(len(v100)),
        "old_meter": pulse_summary(old),
        "v100_meter": pulse_summary(v100),
        "v100_sub_pulse_artifacts": sub_pulse_note,
        "counter_continuous_at_transition": bool(counter_after >= counter_before),
        "counter_value_before": float(counter_before),
        "counter_value_after": float(counter_after),
        "mean_avg_rate_old": float(old["avg_rate"].mean()),
        "mean_avg_rate_v100": float(v100["avg_rate"].mean()),
        "zero_proportion_old": float((old["avg_rate"] == 0).mean()),
        "zero_proportion_v100": float((v100["avg_rate"] == 0).mean()),
    }


def check_v100_grid(v100: pd.DataFrame) -> dict:
    """Confirm the analysis-period subset is itself a clean, gap-free 1-minute grid."""
    diffs = v100["unix_ts"].diff().to_numpy()[1:]
    return {
        "n_rows": len(v100),
        "start": v100["datetime"].iloc[0],
        "start_unix_ts": int(v100["unix_ts"].iloc[0]),
        "end": v100["datetime"].iloc[-1],
        "regular_1min_grid": bool((diffs == SECONDS_PER_MINUTE).all()),
        "duplicate_unix_ts": int(v100["unix_ts"].duplicated().sum()),
        "missing_values": v100[["counter", "avg_rate", "inst_rate"]].isna().sum().to_dict(),
        "span_days": float((v100["unix_ts"].iloc[-1] - v100["unix_ts"].iloc[0]) / 86400),
    }

#%%

def target_stats(target: pd.DataFrame) -> dict:
    """
    Distributional summary of the modelling target (avg_rate) over the
    V100 period: standard descriptive stats, upper-tail quantiles (the
    distribution is expected to be extremely right-skewed / zero-heavy).
    """
    return {
        "describe": {k: float(v) for k, v in target.describe().items()},
        "quantiles": {
            f"p{q * 100:g}": float(target.quantile(q))
            for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999] # recall: quantiles and percentiles sort in ascending order first
        },
        "max": float(target.max()),
        "zero_count": int((target == 0).sum()),
        "zero_proportion": float((target == 0).mean()),
        "mean_daily_volume_L": float(target.sum() / (len(target) / 1440)),
        # "inst_rate_describe": {
        #     k: float(v) for k, v in v100["inst_rate"].describe().items()
        # },
    }

#%%
def event_stats(target: pd.Series) -> dict:
    """
    Characterize discrete water-use "events": contiguous runs of minutes
    with avg_rate > 0, separated by zero-flow gaps.

    This quantifies the event structure referenced in the open horizon
    decision (STATE.md): event duration is one of
    the inputs needed to judge whether a candidate aggregation window (e.g.
    15 minutes) is a sensible unit -- too short a window mostly just
    relocates the same zero-inflation problem to a coarser grid, too long
    blurs distinct events together.
    """
    active = (target > 0).to_numpy()
    values = target.to_numpy(dtype=np.float64)

    starts, ends = segment_runs(active)     # no loop required through this and reduceat
    durations = ends - starts

    volumes = np.add.reduceat(values, starts)

    return {
        "n_events": int(len(starts)),
        "duration_minutes": distribution_summary(durations),
        "volume_per_event_L": distribution_summary(volumes),
        "mean_events_per_day": float(len(starts) / (len(active) / 1440)),
        "active_minute_proportion": float(active.mean()),
    }


def temporal_profiles(v100: pd.DataFrame) -> dict:
    """
    Mean avg_rate and nonzero-fraction, grouped by LOCAL hour-of-day,
    day-of-week, and weekend/weekday. This is the descriptive counterpart
    to the calendar features listed as candidates in AGENTS.md.

    unix_ts is true Unix/UTC time and the
    household is in America/Vancouver, so calendar grouping must use
    datetime_local. Grouping on the UTC-derived hour instead would rotate
    the diurnal profile by 7-8 hours (the local evening peak lands at
    "3 am" UTC) and mislabel day-of-week for evening hours.
    """
    target = v100["avg_rate"]
    out = {"timezone": LOCAL_TZ}

    hour = v100["datetime_local"].dt.hour
    g = target.groupby(hour)
    out["hour_of_day"] = {
        "mean": g.mean().tolist(),
        "nonzero_fraction": (target > 0).groupby(hour).mean().tolist(),
    }

    dow = v100["datetime_local"].dt.dayofweek
    g = target.groupby(dow)
    out["day_of_week"] = {
        "mean": g.mean().tolist(),
        "nonzero_fraction": (target > 0).groupby(dow).mean().tolist(),
    }

    weekend = v100["datetime_local"].dt.dayofweek >= 5
    g = target.groupby(weekend)
    out["weekend"] = {
        "mean": {str(k): float(v) for k, v in g.mean().items()},
        "nonzero_fraction": {
            str(k): float(v) for k, v in (target > 0).groupby(weekend).mean().items()
        },
    }
    return out


def daily_volumes(v100: pd.DataFrame) -> pd.Series:
    """
    Total volume (L) per LOCAL calendar day.
    The first day is dropped.

    tz_localize(None) drops the tz label but keeps the local wall time, so
    flooring to "D" gives one unique key per local calendar day.
    """
    local_day = v100["datetime_local"].dt.tz_localize(None).dt.floor("D")
    return v100["avg_rate"].groupby(local_day).sum().iloc[1:]


def daily_volume_stats(daily: pd.Series) -> dict:
    """
    Distribution of daily volume plus the 5 lowest and 5 highest days.
    Purely descriptive: it shows where atypical days are, not why.
    """
    return {
        "volume_L": distribution_summary(daily.to_numpy()),
        "lowest_days_L": {str(d.date()): float(v) for d, v in daily.nsmallest(5).items()},
        "highest_days_L": {str(d.date()): float(v) for d, v in daily.nlargest(5).items()},
    }


def longest_zero_runs(v100: pd.DataFrame) -> list:
    """
    The 5 longest runs of consecutive zero-flow minutes, with local start/end.
    Ordinary overnight lulls are ~9-10 h; a run far beyond that means nobody
    used water for a long stretch.
    """
    starts, ends = segment_runs((v100["avg_rate"] == 0).to_numpy())
    lengths = ends - starts
    local = v100["datetime_local"]
    return [
        {
            "duration_minutes": int(lengths[i]),
            "start_local": local.iloc[starts[i]],
            "end_local": local.iloc[ends[i] - 1],
        }
        for i in np.argsort(-lengths, kind="stable")[:5]
    ]


def acf_analysis(v100: pd.DataFrame) -> dict:
    """
    Compute two autocorrelation curves at 1-minute resolution, up to
    ACF_MAX_LAG:

      1. "raw": ACF of avg_rate itself.
      2. "indicator": ACF of the binary (avg_rate > 0) occurrence series.

    The comparison matters for interpreting the raw ACF correctly: because
    avg_rate is heavily zero-inflated, a large share of the raw ACF(1) can
    come simply from "a zero minute tends to be followed by another zero
    minute" (consequence of sparsity), rather than from
    genuine short-run persistence in ongoing water-use events. Computing
    the indicator ACF separately lets us see how much of the raw
    autocorrelation is attributable to occurrence patterns alone.
    """
    target = v100["avg_rate"].to_numpy(dtype=np.float64)
    acf = acf_fft(target, ACF_MAX_LAG)
    indicator = (target > 0).astype(np.float64)
    acf_ind = acf_fft(indicator, ACF_MAX_LAG)

    def at_lags(a):
        out = {}
        for lag in CANDIDATE_LAGS:
            out[str(lag)] = float(a[lag]) if lag <= ACF_MAX_LAG else None
        return out

    return {
        "max_lag_minutes": ACF_MAX_LAG,
        "candidate_lags": {str(k): v for k, v in at_lags(acf).items()},
        "indicator_candidate_lags": at_lags(acf_ind),
        "curve": acf.tolist(),
        "indicator_curve": acf_ind.tolist(),
    }


def make_figures(
    df: pd.DataFrame, v100: pd.DataFrame, acf_result: dict, daily: pd.Series
) -> None:
    """ Save all diagnostic figures for this EDA to FIG_DIR """
    target = v100["avg_rate"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(target, bins=100)
    ax.set_xlabel("WHW avg_rate (L/min)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of WHW avg_rate (V100 period)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hist_all.png", dpi=150)
    plt.close(fig)

    # nonzero histogram is more informative than the full histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    nonzero = target[target > 0]
    bin_edges = np.arange(
        PULSE_SIZE_L / 2,
        nonzero.max() + PULSE_SIZE_L,
        PULSE_SIZE_L,
    )
    ax.hist(nonzero, bins=bin_edges)
    ax.set_xlabel("WHW avg_rate (L/min)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of non-zero WHW avg_rate (V100 period)")
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(PULSE_SIZE_L))
    ax.set_xlim(0, nonzero.max() + PULSE_SIZE_L / 2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hist_nonzero.png", dpi=150)
    plt.close(fig)

    # minutely time series plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(v100["datetime_local"], target, linewidth=0.3)
    ax.set_xlabel("Date (local time)")
    ax.set_ylabel("avg_rate (L/min)")
    ax.set_title("Whole-house water consumption — V100 period")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "timeseries_v100.png", dpi=150)
    plt.close(fig)

    # daily totals - level shifts, absences and unusually heavy-use days mroe visible 
    # also 7-day centered mean for display
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily, linewidth=0.8, label="Daily volume")
    ax.plot(
        daily.index,
        daily.rolling(7, center=True).mean(),
        linewidth=1.8,
        color="red",
        label="7-day centered mean",
    )
    ax.set_xlabel("Date (local time)")
    ax.set_ylabel("Volume per day (L)")
    ax.set_title("Whole-house water consumption, daily volume — V100 period")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "daily_volume_v100.png", dpi=150)
    plt.close(fig)

    # First week plot, kinda useless
    week = v100.iloc[: 7 * 1440]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(week["datetime_local"], week["avg_rate"], linewidth=0.8)
    ax.set_xlabel("Date (local time)")
    ax.set_ylabel("avg_rate (L/min)")
    ax.set_title("First week of the V100 period")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_week.png", dpi=150)
    plt.close(fig)

    # THIS is much better
    hour = v100["datetime_local"].dt.hour
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(24), target.groupby(hour).mean(), marker="o")
    axes[0].set_xlabel("Hour of day (local)")
    axes[0].set_ylabel("Mean avg_rate (L/min)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[1].plot(range(24), (target > 0).groupby(hour).mean(), marker="o", color="orange")
    axes[1].set_xlabel("Hour of day (local)")
    axes[1].set_ylabel("Fraction of minutes with consumption")
    axes[1].set_xticks(range(0, 24, 2))
    fig.suptitle("Hour-of-day profile (V100 period, local time)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hour_profile.png", dpi=150)
    plt.close(fig)

    # Interesting Mondays
    dow = v100["datetime_local"].dt.dayofweek
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(7), target.groupby(dow).mean(), marker="o")
    axes[0].set_xticks(range(7), labels)
    axes[0].set_ylabel("Mean avg_rate (L/min)")
    axes[1].plot(range(7), (target > 0).groupby(dow).mean(), marker="o", color="orange")
    axes[1].set_xticks(range(7), labels)
    axes[1].set_ylabel("Fraction of minutes with consumption")
    fig.suptitle("Day-of-week profile (V100 period, local time)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dow_profile.png", dpi=150)
    plt.close(fig)

    # Two rows (raw ACF, indicator ACF) x two columns (full range, 6-hour
    # zoom). Vertical dotted red lines mark the candidate lags from
    # CANDIDATE_LAGS, so it's visually obvious whether a candidate lag sits
    # on a genuine peak or just in the decay tail.
    lags = np.arange(ACF_MAX_LAG + 1)
    curve = np.asarray(acf_result["curve"])
    ind_curve = np.asarray(acf_result["indicator_curve"])
    zoom = 360
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(lags, curve, linewidth=0.8)
    axes[0, 0].set_ylabel("ACF of avg_rate")
    axes[0, 1].plot(lags[: zoom + 1], curve[: zoom + 1], linewidth=0.8)
    axes[0, 1].set_ylabel("ACF of avg_rate")
    axes[0, 1].set_title("Zoom: first 6 hours")
    axes[1, 0].plot(lags, ind_curve, linewidth=0.8, color="orange")
    axes[1, 0].set_ylabel("ACF of (avg_rate > 0)")
    axes[1, 1].plot(lags[: zoom + 1], ind_curve[: zoom + 1], linewidth=0.8, color="orange")
    axes[1, 1].set_ylabel("ACF of (avg_rate > 0)")
    axes[1, 1].set_title("Zoom: first 6 hours")
    for ax in axes.flat:
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (minutes)")
        for lag in CANDIDATE_LAGS:
            if lag <= ACF_MAX_LAG:
                ax.axvline(lag, color="red", linestyle=":", alpha=0.4)
    fig.suptitle("Autocorrelation (V100 period); red lines: candidate lags")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acf.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Orchestrate the full EDA: load -> validate raw data -> analyze the
    meter transition -> restrict to the V100 period -> compute target
    statistics, event structure, daily volumes and zero-flow runs,
    temporal profiles (local time), ACF, and the timezone note -> write the
    cleaned parquet (+ CSV twin), a JSON summary of all numeric results, and figures.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_water()
    raw = check_raw(df)
    transition = analyze_transition(df)

    # Analysis period: the pulse-grid-consistent subset (V100_CLEAN_UNIX_TS
    # onward); the full V100 window from the meter swap is used only for
    # the transition analysis above.
    v100 = df[df["unix_ts"] >= V100_CLEAN_UNIX_TS].reset_index(drop=True).copy()
    grid = check_v100_grid(v100)

    target = v100["avg_rate"]
    stats = target_stats(target)
    events = event_stats(target)
    profiles = temporal_profiles(v100)
    acf = acf_analysis(v100)
    daily = daily_volumes(v100)
    daily_stats = daily_volume_stats(daily)
    timezone = {
        "local_timezone": LOCAL_TZ,
        "unix_ts_is_true_utc": True,
    }

    # The raw ACF/indicator curves are large (10081 floats each) and belong
    # in the figures, not in the JSON summary meant for quick inspection --
    # only the candidate-lag values are kept in the summary dict.
    acf_for_summary = {
        k: v for k, v in acf.items() if k not in ("curve", "indicator_curve")
    }

    summary = {
        "dataset": "AMPds2 Water_WHW.csv",
        "note_unix_ts": (
            "unix_ts is true Unix/UTC time; "
            "stored datetimes are tz-aware UTC; calendar profiles use "
            "America/Vancouver local time"
        ),
        "timezone": timezone,
        "raw": raw,
        "meter_transition": transition,
        "v100_grid": grid,
        "target_stats": stats,
        "events": events,
        "daily_volume": daily_stats,
        "longest_zero_runs": longest_zero_runs(v100),
        "temporal_profiles": profiles,
        "acf": acf_for_summary,
        "artifacts": {
            "parquet": str(PARQUET_PATH.relative_to(REPO_ROOT)),
            "csv": str(CSV_PATH.relative_to(REPO_ROOT)),
            "figures_dir": str(FIG_DIR.relative_to(REPO_ROOT)),
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(to_serializable(summary), f, indent=2)

    out = v100[["unix_ts", "datetime_local", "counter", "avg_rate"]].copy()
    out = out.set_index("unix_ts")  # unique, strictly monotone UTC master grid
    out.to_parquet(PARQUET_PATH)
    out.to_csv(CSV_PATH)  # identical content, for direct inspection

    make_figures(df, v100, acf, daily)

    print(json.dumps(to_serializable({k: summary[k] for k in ("raw", "meter_transition", "v100_grid", "target_stats", "events", "daily_volume", "longest_zero_runs", "acf")}), indent=2))
    print(f"\nWrote {PARQUET_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()