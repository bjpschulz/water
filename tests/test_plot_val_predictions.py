import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from plot_val_predictions import select_validation_window


def test_select_validation_week_and_day_across_dst_transition():
    local_start = pd.Timestamp("2024-03-04 00:00", tz="America/Vancouver")
    local_end = pd.Timestamp("2024-03-11 00:00", tz="America/Vancouver")
    local_times = pd.date_range(local_start, local_end, freq="min", inclusive="left")
    predictions = pd.DataFrame(
        {"unix_ts": [int(timestamp.timestamp()) for timestamp in local_times]}
    )

    week, week_local_times = select_validation_window(predictions, 1, None)
    sunday, sunday_local_times = select_validation_window(predictions, 1, 7)

    assert len(week) == 7 * 24 * 60 - 60
    assert week_local_times[0] == local_start
    assert week_local_times[-1] == local_end - pd.Timedelta(minutes=1)
    assert len(sunday) == 23 * 60
    assert (sunday_local_times.dayofweek == 6).all()
