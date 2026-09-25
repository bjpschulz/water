import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from build_features import LAGS, MAX_LAG, add_calendar, add_lags


def test_add_lags():
    n = MAX_LAG + 5  # longer than the largest lag, so every column gets a real comparison
    source = pd.DataFrame(
        {"avg_rate": np.arange(n, dtype=float)},
        index=pd.Index(np.arange(n), name="unix_ts"),
    )

    result = add_lags(source)

    pd.testing.assert_series_equal(result["avg_rate"], source["avg_rate"])
    expected = source["avg_rate"].to_numpy()
    for k in LAGS:
        values = result[f"lag_{k}"].to_numpy()
        assert np.array_equal(values[k:], expected[:-k]), f"lag_{k} is not a backward shift by {k}"
        assert result[f"lag_{k}"].isna().sum() == k, f"lag_{k} should have exactly {k} warm-up NaNs"


def test_add_calendar():
    source = pd.DataFrame(
        {
            "datetime_local": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 12:00:00",
                    "2024-01-06 23:00:00",
                ],
                utc=True,
            ).tz_convert("America/Vancouver")
        }
    )

    result = add_calendar(source)

    assert result["hour"].tolist() == [16, 4, 15]
    assert result["dow_local"].tolist() == [6, 0, 5]
    assert result["weekend"].tolist() == [True, False, True]
    np.testing.assert_allclose(
        result["hour_sin"], np.sin(2 * np.pi * result["hour"] / 24)
    )
    np.testing.assert_allclose(
        result["hour_cos"], np.cos(2 * np.pi * result["hour"] / 24)
    )
    np.testing.assert_allclose(
        result["dow_sin"], np.sin(2 * np.pi * result["dow_local"] / 7)
    )
    np.testing.assert_allclose(
        result["dow_cos"], np.cos(2 * np.pi * result["dow_local"] / 7)
    )


# def test_hour_embedding_wraps_at_midnight():
#     # hour 23 and hour 0 should sit close together on the unit circle,
#     # not at opposite ends -- that's the whole point of the cyclic encoding.
#     source = pd.DataFrame(
#         {
#             "datetime_local": pd.to_datetime(
#                 ["2024-01-01 23:00:00", "2024-01-02 00:00:00", "2024-01-01 11:00:00"]
#             ).tz_localize("America/Vancouver")
#         }
#     )
#     h23, h0, h11 = add_calendar(source).itertuples()

#     dist_across_midnight = np.hypot(h23.hour_sin - h0.hour_sin, h23.hour_cos - h0.hour_cos)
#     dist_half_day_apart = np.hypot(h23.hour_sin - h11.hour_sin, h23.hour_cos - h11.hour_cos)
#     assert dist_across_midnight < dist_half_day_apart