"""Reproducible EDA for the AMPds2 whole-house water (WHW) data.

Run: uv run python scripts/eda_whw.py

Outputs:
  data/processed/whw_v100.parquet   cleaned V100-meter period (1-min grid)
  results/eda_whw/summary.json      all key numbers
  results/eda_whw/figs/*.png        figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Water_WHW.csv"
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "whw_v100.parquet"
OUT_DIR = REPO_ROOT / "results" / "eda_whw"
FIG_DIR = OUT_DIR / "figs"

V100_UNIX_TS = 1_342_287_780
CANDIDATE_LAGS = [1, 5, 15, 30, 60, 1440, 10080]
ACF_MAX_LAG = 10080
SECONDS_PER_MINUTE = 60


def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def load_water() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["unix_ts"], unit="s")
    return df.sort_values("unix_ts").reset_index(drop=True)


def check_raw(df: pd.DataFrame) -> dict:
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
    counter_vs_rate = (df["counter"].diff() - df["avg_rate"]).abs()
    out["max_abs_counter_diff_minus_avg_rate"] = float(counter_vs_rate.iloc[1:].max())
    return out


def analyze_transition(df: pd.DataFrame) -> dict:
    old = df[df["unix_ts"] < V100_UNIX_TS]
    v100 = df[df["unix_ts"] >= V100_UNIX_TS]

    def pulse_summary(period: pd.DataFrame) -> dict:
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

    v100_increments = v100["counter"].diff()
    sub_pulse = v100_increments[(v100_increments > 0) & (v100_increments < 0.5)]
    sub_pulse_note = {
        "count": int(len(sub_pulse)),
        "timestamps": v100.loc[sub_pulse.index, "datetime"].tolist(),
        "note": (
            "One-off counter settling onto the 0.5 L pulse grid shortly after the "
            "meter swap; all other V100 increments are multiples of 0.5 L."
        ),
    }

    return {
        "v100_unix_ts": V100_UNIX_TS,
        "v100_datetime": df["datetime"].iloc[transition_row],
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
    diffs = v100["unix_ts"].diff().to_numpy()[1:]
    return {
        "n_rows": len(v100),
        "start": v100["datetime"].iloc[0],
        "end": v100["datetime"].iloc[-1],
        "regular_1min_grid": bool((diffs == SECONDS_PER_MINUTE).all()),
        "duplicate_unix_ts": int(v100["unix_ts"].duplicated().sum()),
        "missing_values": v100[["counter", "avg_rate", "inst_rate"]].isna().sum().to_dict(),
        "span_days": float((v100["unix_ts"].iloc[-1] - v100["unix_ts"].iloc[0]) / 86400),
    }


def target_stats(v100: pd.DataFrame) -> dict:
    target = v100["avg_rate"]
    return {
        "describe": {k: float(v) for k, v in target.describe().items()},
        "quantiles": {
            f"p{q * 100:g}": float(target.quantile(q))
            for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
        },
        "max": float(target.max()),
        "zero_count": int((target == 0).sum()),
        "zero_proportion": float((target == 0).mean()),
        "mean_daily_volume_L": float(target.sum() / (len(target) / 1440)),
        "inst_rate_describe": {
            k: float(v) for k, v in v100["inst_rate"].describe().items()
        },
    }


def event_stats(target: pd.Series) -> dict:
    active = (target > 0).to_numpy()
    values = target.to_numpy(dtype=np.float64)
    d = np.diff(active.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if active[0]:
        starts = np.concatenate([[0], starts])
    if active[-1]:
        ends = np.concatenate([ends, [len(active)]])
    durations = ends - starts
    volumes = np.add.reduceat(values, starts)

    def dist(x):
        x = np.asarray(x, dtype=np.float64)
        return {
            "count": int(len(x)),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)),
            "max": float(x.max()),
        }

    return {
        "n_events": int(len(starts)),
        "duration_minutes": dist(durations),
        "volume_per_event_L": dist(volumes),
        "mean_events_per_day": float(len(starts) / (len(active) / 1440)),
        "active_minute_proportion": float(active.mean()),
    }


def temporal_profiles(v100: pd.DataFrame) -> dict:
    target = v100["avg_rate"]
    out = {}

    hour = v100["datetime"].dt.hour
    g = target.groupby(hour)
    out["hour_of_day"] = {
        "mean": g.mean().tolist(),
        "nonzero_fraction": (target > 0).groupby(hour).mean().tolist(),
    }

    dow = v100["datetime"].dt.dayofweek
    g = target.groupby(dow)
    out["day_of_week"] = {
        "mean": g.mean().tolist(),
        "nonzero_fraction": (target > 0).groupby(dow).mean().tolist(),
    }

    weekend = v100["datetime"].dt.dayofweek >= 5
    g = target.groupby(weekend)
    out["weekend"] = {
        "mean": {str(k): float(v) for k, v in g.mean().items()},
        "nonzero_fraction": {
            str(k): float(v) for k, v in (target > 0).groupby(weekend).mean().items()
        },
    }
    return out


def acf_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(f * f.conj(), n=nfft)[: max_lag + 1]
    denom = float((x * x).sum())
    return acov / denom if denom > 0 else np.zeros(max_lag + 1)


def acf_analysis(v100: pd.DataFrame) -> dict:
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


def make_figures(df: pd.DataFrame, v100: pd.DataFrame, acf_result: dict) -> None:
    target = v100["avg_rate"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(target, bins=100)
    ax.set_xlabel("WHW avg_rate (L/min)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of WHW avg_rate (V100 period)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hist_all.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(target[target > 0], bins=100)
    ax.set_xlabel("WHW avg_rate (L/min)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of non-zero WHW avg_rate (V100 period)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hist_nonzero.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(v100["datetime"], target, linewidth=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("avg_rate (L/min)")
    ax.set_title("Whole-house water consumption — V100 period")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "timeseries_v100.png", dpi=150)
    plt.close(fig)

    week = v100.iloc[: 7 * 1440]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(week["datetime"], week["avg_rate"], linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("avg_rate (L/min)")
    ax.set_title("First week of the V100 period")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_week.png", dpi=150)
    plt.close(fig)

    transition_dt = v100["datetime"].iloc[0]
    window = df[
        (df["datetime"] >= transition_dt - pd.Timedelta(hours=2))
        & (df["datetime"] <= transition_dt + pd.Timedelta(hours=2))
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.step(window["datetime"], window["avg_rate"], where="post", linewidth=0.8)
    ax.axvline(transition_dt, color="red", linestyle="--", label="V100 transition")
    ax.set_xlabel("Time")
    ax.set_ylabel("avg_rate (L/min)")
    ax.set_title("Meter transition (old meter has coarser pulses)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "transition_zoom.png", dpi=150)
    plt.close(fig)

    hour = v100["datetime"].dt.hour
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(24), target.groupby(hour).mean(), marker="o")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean avg_rate (L/min)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[1].plot(range(24), (target > 0).groupby(hour).mean(), marker="o", color="orange")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Fraction of minutes with consumption")
    axes[1].set_xticks(range(0, 24, 2))
    fig.suptitle("Hour-of-day profile (V100 period)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hour_profile.png", dpi=150)
    plt.close(fig)

    dow = v100["datetime"].dt.dayofweek
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(7), target.groupby(dow).mean(), marker="o")
    axes[0].set_xticks(range(7), labels)
    axes[0].set_ylabel("Mean avg_rate (L/min)")
    axes[1].plot(range(7), (target > 0).groupby(dow).mean(), marker="o", color="orange")
    axes[1].set_xticks(range(7), labels)
    axes[1].set_ylabel("Fraction of minutes with consumption")
    fig.suptitle("Day-of-week profile (V100 period)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dow_profile.png", dpi=150)
    plt.close(fig)

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
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_water()
    raw = check_raw(df)
    transition = analyze_transition(df)

    v100 = df[df["unix_ts"] >= V100_UNIX_TS].reset_index(drop=True).copy()
    grid = check_v100_grid(v100)

    target = v100["avg_rate"]
    stats = target_stats(v100)
    events = event_stats(target)
    profiles = temporal_profiles(v100)
    acf = acf_analysis(v100)

    acf_for_summary = {
        k: v for k, v in acf.items() if k not in ("curve", "indicator_curve")
    }

    summary = {
        "dataset": "AMPds2 Water_WHW.csv",
        "note_unix_ts": "unix_ts interpreted as Unix time; datetimes are UTC-derived and naive",
        "raw": raw,
        "meter_transition": transition,
        "v100": grid,
        "target": stats,
        "events": events,
        "temporal_profiles": profiles,
        "acf": acf_for_summary,
        "artifacts": {
            "parquet": str(PARQUET_PATH.relative_to(REPO_ROOT)),
            "figures_dir": str(FIG_DIR.relative_to(REPO_ROOT)),
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(to_serializable(summary), f, indent=2)

    out = v100[["unix_ts", "counter", "avg_rate", "inst_rate"]].copy()
    out.index = v100["datetime"]
    out.index.name = "datetime"
    out.to_parquet(PARQUET_PATH)

    make_figures(df, v100, acf)

    print(json.dumps(to_serializable({k: summary[k] for k in ("raw", "meter_transition", "v100", "target", "events", "acf")}), indent=2))
    print(f"\nWrote {PARQUET_PATH}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
