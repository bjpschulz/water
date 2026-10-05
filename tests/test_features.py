import numpy as np
import pandas as pd

from core.config import LAGS
from core.features import add_calendar, add_lags


def _series(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"avg_rate": np.arange(n, dtype=float)},
        index=pd.Index(np.arange(n), name="unix_ts"),
    )


def test_add_lags_is_backward_shift():
    source = _series(max(LAGS) + 5)  # longer than the largest lag, so every column is compared
    result = add_lags(source)

    pd.testing.assert_series_equal(result["avg_rate"], source["avg_rate"])
    expected = source["avg_rate"].to_numpy()
    for k in LAGS:
        values = result[f"lag_{k}"].to_numpy()
        assert np.array_equal(values[k:], expected[:-k]), f"lag_{k} is not a backward shift by {k}"
        assert result[f"lag_{k}"].isna().sum() == k, f"lag_{k} should have exactly {k} warm-up NaNs"


def test_lags_do_not_see_the_future():
    # Changing the target from row t onward must not change any feature before row t.
    source = _series(max(LAGS) + 50)
    t = len(source) - 20
    changed = source.copy()
    changed.iloc[t:, 0] = -999.0

    before = add_lags(source).drop(columns="avg_rate").iloc[: t + 1]
    after = add_lags(changed).drop(columns="avg_rate").iloc[: t + 1]
    pd.testing.assert_frame_equal(before, after)


def test_add_calendar_uses_local_time():
    source = pd.DataFrame(
        {
            "datetime_local": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 12:00:00", "2024-01-06 23:00:00"],
                utc=True,
            ).tz_convert("America/Vancouver")
        }
    )
    result = add_calendar(source)

    assert result["hour"].tolist() == [16, 4, 15]
    assert result["dow_local"].tolist() == [6, 0, 5]
    assert result["weekend"].tolist() == [True, False, True]
    np.testing.assert_allclose(result["hour_sin"], np.sin(2 * np.pi * result["hour"] / 24))
    np.testing.assert_allclose(result["hour_cos"], np.cos(2 * np.pi * result["hour"] / 24))
    np.testing.assert_allclose(result["dow_sin"], np.sin(2 * np.pi * result["dow_local"] / 7))
    np.testing.assert_allclose(result["dow_cos"], np.cos(2 * np.pi * result["dow_local"] / 7))
