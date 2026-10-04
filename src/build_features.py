"""
Build the modelling feature table for the WHW V100-period series.

Every feature is a deterministic row-wise transform -- backward shifts of
the target (lags) or pure functions of the timestamp (calendar) -- so
building the table on the full series *before* splitting is leakage-safe.

Feature groups (justifications measured in output/initial_eda/summary.json):

  lags       lag_1 .. lag_15, lag_30, lag_45, lag_60, lag_1440, lag_10080
             The occurrence lag (occ_lag_1) is NOT built yet.
  calendar   hour, hour_sin, hour_cos, dow_local, dow_sin, dow_cos, weekend
             Raw integers and cyclic encodings are both stored; since different
             encoding suits different model classes.

Warm-up NaN policy: the first rows where lags are undefined (10,080 rows for lag_10080)
are KEPT. Drop them at train time if using a linear model;
and also for tree models if they are to be compared.

Run: uv run python src/build_features.py

Outputs:
  output/build_features/whw_v100_with_features.parquet feature table (index unix_ts)
  output/build_features/whw_v100_with_features.csv     identical content, for skimming
"""

#%%
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ts_utils import LAGS

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = REPO_ROOT / "output" / "initial_eda" / "whw_v100.parquet"
OUT_DIR = REPO_ROOT / "output" / "build_features"
OUT_PARQUET = OUT_DIR / "whw_v100_with_features.parquet"
OUT_CSV = OUT_DIR / "whw_v100_with_features.csv"  # twin, for quick skimming

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
    """ Runtime data-integrity checks.
    Catches pipeline-level corruption on real data that a synthetic fixed unit-test can't. """
    checks = {}

    checks["target_identical_to_input"] = bool(df["avg_rate"].equals(src["avg_rate"]))
    checks["row_count_identical_to_input"] = len(df) == len(src)

    expected = src["avg_rate"].to_numpy()
    for k in LAGS:
        values = df[f"lag_{k}"].to_numpy()
        values_match = np.array_equal(values[k:], expected[:-k], equal_nan=True)
        nan_count_ok = int(df[f"lag_{k}"].isna().sum()) == k
        checks[f"lag_{k}_correct"] = bool(values_match and nan_count_ok)

    return checks


#%%

def main() -> None:
    """ Build the feature table, check and save it. """
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

if __name__ == "__main__":
    main()
