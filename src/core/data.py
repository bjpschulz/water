"""Load the feature table and split it chronologically into train/valid/test."""

import numpy as np
import pandas as pd

from core.config import FEATURES_PARQUET, LOCAL_TZ, TARGET

VALID_WEEKS = 13
TEST_WEEKS = 13


def load_features() -> pd.DataFrame:
    """Load the feature table (index unix_ts)."""
    return pd.read_parquet(FEATURES_PARQUET)


def split_series(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Chronological train/valid/test split (~70/15/15).

    Boundaries are Monday 00:00 local time, counted back from the last Monday
    midnight in the series, so valid/test cover full weekly cycles (DST adds or
    removes an hour of rows). Train excludes the leading lag warm-up rows (NaN
    features); the three slices are contiguous and cover the complete-case region.
    """
    local = df["datetime_local"]
    mondays = np.flatnonzero(
        (local.dt.dayofweek == 0) & (local.dt.hour == 0) & (local.dt.minute == 0)
    )
    test_pos = int(mondays[-(TEST_WEEKS + 1)])
    valid_pos = int(mondays[-(TEST_WEEKS + VALID_WEEKS + 1)])
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    warmup = int(df[lag_cols].isna().any(axis=1).sum())
    return {
        "train": df.iloc[warmup:valid_pos],
        "valid": df.iloc[valid_pos:test_pos],
        "test": df.iloc[test_pos:],
    }


def describe_splits(splits: dict[str, pd.DataFrame]) -> dict:
    """Rows and local-time start/end of each split, for the summaries."""
    out = {}
    for name, frame in splits.items():
        local = pd.to_datetime(frame.index[[0, -1]], unit="s", utc=True).tz_convert(LOCAL_TZ)
        out[name] = {"rows": len(frame), "start_local": local[0], "end_local": local[1]}
    return out


def xy(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) as float64 arrays for a given feature list."""
    X = frame[features].astype(np.float64).to_numpy()
    y = frame[TARGET].to_numpy(dtype=np.float64)
    return X, y
