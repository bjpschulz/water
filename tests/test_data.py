import numpy as np
import pandas as pd

from core.data import TEST_WEEKS, VALID_WEEKS, split_series


def _frame(weeks: int = 40, warmup: int = 5) -> pd.DataFrame:
    """Hourly synthetic table with lag NaN warm-up rows (a Wednesday start)."""
    ts = pd.date_range("2013-01-02", periods=weeks * 168, freq="h", tz="UTC")
    df = pd.DataFrame({"avg_rate": np.arange(len(ts), dtype=float), "lag_1": 1.0}, index=ts.view("int64") // 10**9)
    df.index.name = "unix_ts"
    df["datetime_local"] = ts.tz_convert("America/Vancouver")
    df.loc[df.index[:warmup], "lag_1"] = np.nan
    return df


def test_split_is_chronological():
    df = _frame()
    s = split_series(df)
    joined = pd.concat([s["train"], s["valid"], s["test"]])
    assert joined.index.is_monotonic_increasing and joined.index.is_unique
    assert len(joined) == len(df) - 5  # only the warm-up rows are dropped
    assert not joined["lag_1"].isna().any()
    assert s["train"].index[-1] < s["valid"].index[0] and s["valid"].index[-1] < s["test"].index[0]


def test_split_starts_on_monday():
    s = split_series(_frame())
    for name, weeks in (("valid", VALID_WEEKS), ("test", TEST_WEEKS)):
        first = s[name]["datetime_local"].iloc[0]
        assert (first.dayofweek, first.hour, first.minute) == (0, 0, 0)
    assert abs(len(s["valid"]) - VALID_WEEKS * 168) <= 1  # DST may shift by an hour
