"""
Build the modelling feature table for the cleaned V100-period series.

Columns: datetime_local, counter, avg_rate, then lag_1..lag_15, lag_30, lag_45,
lag_60, lag_1440, lag_10080 and the calendar features hour, hour_sin, hour_cos,
dow_local, dow_sin, dow_cos, weekend (see core/features.py). The leading rows
with undefined lags are kept; core.data.split_series excludes them from train.

Run: uv run python src/build_features.py
Output: output/build_features/whw_v100_with_features.parquet (index unix_ts)
        and an identical .csv twin, for skimming
"""

import pandas as pd

from core.config import FEATURES_PARQUET, LAG_COLS, V100_PARQUET
from core.features import add_calendar, add_lags, check_features

CALENDAR_COLS = ["hour", "hour_sin", "hour_cos", "dow_local", "dow_sin", "dow_cos", "weekend"]


def load_v100() -> pd.DataFrame:
    """Load the cleaned V100-period series and check it is a gap-free 1-minute grid."""
    df = pd.read_parquet(V100_PARQUET)
    checks = {
        "index_name_is_unix_ts": df.index.name == "unix_ts",
        "index_unique": bool(df.index.is_unique),
        "index_monotone_increasing": bool(df.index.is_monotonic_increasing),
        "regular_1min_grid": bool((df.index.to_series().diff().dropna() == 60).all()),
        "no_missing_values": bool(df[["avg_rate", "counter", "datetime_local"]].notna().all().all()),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Input artifact failed integrity checks: {failed}")
    return df


def main() -> None:
    """Build the feature table, check it and save it."""
    src = load_v100()
    df = add_calendar(add_lags(src))

    failed = [k for k, ok in check_features(src, df).items() if not ok]
    if failed:
        raise ValueError(f"Feature-table self-checks failed: {failed}")

    df = df[["datetime_local", "counter", "avg_rate"] + LAG_COLS + CALENDAR_COLS]
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET)
    df.to_csv(FEATURES_PARQUET.with_suffix(".csv"))
    print(f"Saved {FEATURES_PARQUET} ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
