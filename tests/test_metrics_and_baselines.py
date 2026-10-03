import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baselines import calculate_baselines
from ts_utils import mase, mase_scale


def test_mase_scale_is_mean_abs_one_step_difference():
    assert mase_scale([0, 1, 1, 3]) == pytest.approx((1 + 0 + 2) / 3)
    with pytest.raises(ValueError):
        mase_scale([2, 2, 2])  # zero scale is rejected


def test_mase_is_mae_over_scale():
    assert mase([0, 2], [1, 0], scale=1.5) == pytest.approx(1.5 / 1.5)


def test_baselines_use_train_only_statistics_and_past_lags():
    def part(y, lag):
        return pd.DataFrame({"avg_rate": y, "lag_1": lag, "lag_1440": lag, "lag_10080": lag},
                            index=pd.Index(range(len(y)), name="unix_ts"))

    train = part([0.0, 2.0, 0.0, 2.0], [0.0] * 4)
    valid = part([10.0, 10.0], [7.0, 7.0])  # huge valid values must not leak into the train mean
    summary, preds = calculate_baselines({"train": train, "valid": valid})
    assert summary["train_target_mean"] == pytest.approx(1.0)
    assert summary["mase_scale"] == pytest.approx(2.0)
    assert np.allclose(preds["always_mean"], 1.0)
    assert np.allclose(preds["persistence"], 7.0)  # taken from the precomputed lag column
