"""
Build the modelling feature table for the WHW V100-period series.

Every feature is a deterministic row-wise transform -- backward shifts of
the target (lags) or pure functions of the timestamp (calendar) -- so
building the table on the full series *before* splitting is leakage-safe.

Feature groups (justifications measured in results/eda_whw/summary.json):

  lags       lag_1, lag_5, lag_15, lag_30, lag_60, lag_1440, lag_10080
             (avg_rate shifted by k minutes; k <= ACF decay region and the
             24h / 168h ACF bumps). The occurrence lag (occ_lag_1) is
             deliberately NOT built yet.
  calendar   hour, hour_sin, hour_cos, dow_local, dow_sin, dow_cos, weekend
             -- computed from datetime_local (America/Vancouver), never from
             the UTC-derived time (settled fact, AGENTS.md). Raw integers and
             cyclic encodings are both stored; since different encoding suits 
             different model classes.

Warm-up NaN policy: the first rows where lags are undefined (10,080 rows for lag_10080)
are KEPT. Drop them at train time if using a linear model;
and also for tree models if they are to be compared.

Run: uv run python src/build_features.py

Outputs:
  data/processed/whw_features_v1.parquet   feature table (index unix_ts)
  data/processed/whw_features_v1.csv       identical content, for skimming
  results/build_features/summary.json      row/column inventory, NaN counts,
                                           and the results of all self-checks
"""

#%%
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ts_utils import to_serializable

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = REPO_ROOT / "data" / "processed" / "whw_v100.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "processed" / "whw_features_v1.parquet"
OUT_CSV = REPO_ROOT / "data" / "processed" / "whw_features_v1.csv"  # twin, for quick skimming
OUT_DIR = REPO_ROOT / "results" / "build_features"

# Lag set from AGENTS.md ("Temporal autocorrelation" / Stage 3), in minutes.
LAGS = [1, 5, 15, 30, 60, 1440, 10080]
MAX_LAG = max(LAGS)

#%%

def load_v100() -> pd.DataFrame:
    """ Load the V100-period series and run integrity checks on it."""
    df = pd.read_parquet(IN_PARQUET)    # index is unix_ts, columns are counter, avg_rate, datetime_local
    checks = {
        "index_name_is_unix_ts": df.index.name == "unix_ts",
        "index_unique": bool(df.index.is_unique),
        "index_monotone_increasing": bool(df.index.is_monotonic_increasing),
        "regular_1min_grid": bool(
            (df.index.to_series().diff().dropna() == 60).all()
        ),
        "no_missing_avg_rate": bool(df["avg_rate"].notna().all()),
        "no_missing_counter": bool(df["counter"].notna().all()),
        "no_missing_datetime_local": bool(df["datetime_local"].notna().all()),
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Input artifact failed integrity checks: {failed}")
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """ Backward-shifted target values: lag_k at row t is avg_rate at t - k minutes. """
    out = df.copy()
    target = out["avg_rate"]
    for k in LAGS:
        out[f"lag_{k}"] = target.shift(k)   # shift only looks at earlier rows, so no leakage
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features from datetime_local (America/Vancouver) -- using the
    UTC-derived hour instead would rotate the diurnal profile by 7-8 hours.

    Both encodings are stored because they suit different model families:
    raw integers (hour, dow_local) for tree models, cyclic sin/cos pairs for linear models.
    """
    out = df.copy()
    local = out["datetime_local"]
    hour = local.dt.hour
    dow = local.dt.dayofweek  # Monday=0 .. Sunday=6

    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_local"] = dow
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["weekend"] = dow >= 5
    return out


def validate_features(src: pd.DataFrame, df: pd.DataFrame) -> dict:
    """ Self-checks on the built table. """
    checks = {}

    target = df["avg_rate"]
    checks["target_identical_to_input"] = bool(target.equals(src["avg_rate"]))
    checks["row_count_identical_to_input"] = len(df) == len(src)

    for k in LAGS:
        col = df[f"lag_{k}"]
        values = col.to_numpy()
        expected = src["avg_rate"].to_numpy()
        checks[f"lag_{k}_values_match_shift"] = bool(
            np.array_equal(values[k:], expected[:-k], equal_nan=True)
        )
        checks[f"lag_{k}_nan_count_equals_{k}"] = int(col.isna().sum()) == k

    checks["hour_in_range"] = bool(df["hour"].between(0, 23).all())
    checks["dow_in_range"] = bool(df["dow_local"].between(0, 6).all())
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        checks[f"{col}_in_unit_range"] = bool(df[col].between(-1.0, 1.0).all())
    checks["weekend_matches_dow"] = bool(
        (df["weekend"] == df["datetime_local"].dt.day_name().isin(["Saturday", "Sunday"])).all()
    )

    # Cyclic continuity spot-check: the hour embedding must wrap around,
    # i.e. hour 23 and hour 0 map to nearby points on the unit circle.
    h23 = df.loc[df["hour"] == 23, ["hour_sin", "hour_cos"]].iloc[0]
    h0 = df.loc[df["hour"] == 0, ["hour_sin", "hour_cos"]].iloc[0]
    checks["hour_embedding_wraps_at_midnight"] = bool(
        np.hypot(h23["hour_sin"] - h0["hour_sin"], h23["hour_cos"] - h0["hour_cos"])
        < np.hypot(
            df.loc[df["hour"] == 12, "hour_sin"].iloc[0] - h0["hour_sin"],
            df.loc[df["hour"] == 12, "hour_cos"].iloc[0] - h0["hour_cos"],
        )
    )

    return checks


#%%

def main() -> None:
    """ Build the feature table and save it, witha JSON summary and self-checks. """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    src = load_v100()
    df = add_lags(src)
    df = add_calendar(df)

    checks = validate_features(src, df)
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Feature-table self-checks failed: {failed}")

    feature_cols = [f"lag_{k}" for k in LAGS] + [
        "hour", "hour_sin", "hour_cos",
        "dow_local", "dow_sin", "dow_cos",
        "weekend",
    ]
    df = df[["datetime_local", "counter", "avg_rate"] + feature_cols]

    df.to_parquet(OUT_PARQUET)
    df.to_csv(OUT_CSV)  # identical content, for direct inspection

    summary = {
        "input_artifact": str(IN_PARQUET.relative_to(REPO_ROOT)),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "start_unix_ts": int(df.index[0]),
        "end_unix_ts": int(df.index[-1]),
        "start_local": df["datetime_local"].iloc[0],
        "end_local": df["datetime_local"].iloc[-1],
        "lags_minutes": LAGS,
        "occurrence_lag_built": False,
        "warmup_rows_kept": MAX_LAG,
        "warmup_policy": (
            f"The first {MAX_LAG} rows have NaN lags and are kept so the table "
            "spans the full grid; the chronological split must place this tail "
            "inside the training region (STATE.md known issue)."
        ),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "nan_counts": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "checks": checks,
        "artifacts": {
            "parquet": str(OUT_PARQUET.relative_to(REPO_ROOT)),
            "csv": str(OUT_CSV.relative_to(REPO_ROOT)),
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(to_serializable(summary), f, indent=2)

    print(json.dumps(to_serializable(summary), indent=2))
    print(f"\nWrote {OUT_PARQUET}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
