import numpy as np
import pandas as pd
import pytest

from core.evaluation import score
from core.naive import naive_forecasts


def test_naive_forecasts_use_given_mean_and_past_lags():
    frame = pd.DataFrame({"avg_rate": [10.0, 10.0], "lag_1": [7.0, 8.0], "lag_1440": [1.0, 2.0], "lag_10080": [3.0, 4.0]})
    preds = naive_forecasts(frame, fit_mean=1.0)  # the mean comes from earlier data, not from `frame`
    assert np.allclose(preds["always_zero"], 0.0)
    assert np.allclose(preds["always_mean"], 1.0)
    assert np.allclose(preds["persistence"], [7.0, 8.0])
    assert np.allclose(preds["seasonal_naive_daily"], [1.0, 2.0])
    assert np.allclose(preds["seasonal_naive_weekly"], [3.0, 4.0])


def test_score():
    result = score(np.array([0.0, 0.0, 2.0, 4.0]), np.array([0.0, 1.0, 2.0, 0.0]))
    assert result["mae"] == pytest.approx(5 / 4)
    assert result["rmse"] == pytest.approx(np.sqrt(17 / 4))
    assert result["zero_prediction_share"] == pytest.approx(0.5)
