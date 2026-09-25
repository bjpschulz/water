from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = REPO_ROOT / "data" / "processed" / "whw_features_v1.parquet"

VALID_WEEKS = 13
TEST_WEEKS = 13


def load_features() -> pd.DataFrame:
    """Load the feature table (index unix_ts)."""
    return pd.read_parquet(IN_PARQUET)


def split_series(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Chronological train/valid/test split (~70/15/15).

    Boundaries: Monday 00:00 America/Vancouver, counted back from the last
    Monday midnight in the series, so valid/test cover full weekly cycles.
    This keeps weekly seasonality and weekly baselines comparable across
    splits; DST may make a local week contain slightly more or fewer UTC
    minute rows, which is expected. Splits remain chronological, not random.
    Train excludes the leading lag warm-up rows (NaN features); the three
    slices are contiguous and concatenate to the complete-case region.
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


if __name__ == "__main__":
    features = load_features()
    splits = split_series(features)
    lag_cols = [c for c in features.columns if c.startswith("lag_")]
    warmup = int(features[lag_cols].isna().any(axis=1).sum())
    complete_case_rows = len(features) - warmup

    print("split lengths:", {k: len(v) for k, v in splits.items()})
    print("feature rows:", len(features))
    print("lag warm-up rows:", warmup)
    print("complete-case rows:", complete_case_rows)
    print("split rows:", sum(len(v) for v in splits.values()))
    print("coverage check:", sum(len(v) for v in splits.values()) == complete_case_rows)

    for name, frame in splits.items():
        start = pd.to_datetime(frame.index[0], unit="s", utc=True)
        end = pd.to_datetime(frame.index[-1], unit="s", utc=True)
        local_start = start.tz_convert("America/Vancouver")
        local_end = end.tz_convert("America/Vancouver")
        print(
            f"{name} boundaries: {local_start} -> {local_end}; "
            f"UTC span minutes={(end - start).total_seconds() / 60:.0f}"
        )
