"""
Feature construction: lags of the target and calendar features in local time.

Every feature is a deterministic row-wise transform (a backward shift of the
target or a function of the timestamp), so the table can be built on the full
series before splitting without leakage.
"""

import numpy as np
import pandas as pd

from core.config import LAGS, TARGET


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-shifted target values: lag_k at row t is avg_rate at t - k minutes."""
    out = df.copy()
    for k in LAGS:
        out[f"lag_{k}"] = out[TARGET].shift(k)  # shift only looks at earlier rows
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features from datetime_local (America/Vancouver); the UTC hour would
    rotate the diurnal profile by 7-8 hours. Raw integers suit tree models,
    cyclic sin/cos pairs suit linear models, so both are stored.
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


def check_features(src: pd.DataFrame, df: pd.DataFrame) -> dict[str, bool]:
    """Integrity checks of the built table against its source series (target unchanged, exact lags)."""
    checks = {
        "target_identical_to_input": bool(df[TARGET].equals(src[TARGET])),
        "row_count_identical_to_input": len(df) == len(src),
    }
    expected = src[TARGET].to_numpy()
    for k in LAGS:
        values = df[f"lag_{k}"].to_numpy()
        values_match = np.array_equal(values[k:], expected[:-k], equal_nan=True)
        nan_count_ok = int(df[f"lag_{k}"].isna().sum()) == k
        checks[f"lag_{k}_correct"] = bool(values_match and nan_count_ok)
    return checks
