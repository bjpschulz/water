"""The five naive baselines, as fixed rules (no fitting beyond a mean)."""

import numpy as np
import pandas as pd

# Name -> lag column. lag_k[t] == avg_rate[t-k] is guaranteed by build_features,
# so the lag baselines are read from the feature table, not re-derived.
LAG_BASELINES = {
    "persistence": "lag_1",
    "seasonal_naive_daily": "lag_1440",
    "seasonal_naive_weekly": "lag_10080",
}

DEFINITIONS = {
    "always_zero": "Predict 0 L/min for every row",
    "always_mean": "Predict the mean avg_rate of the fitting data",
    "persistence": "Predict avg_rate from 1 UTC-minute earlier",
    "seasonal_naive_daily": "Predict avg_rate from 1,440 UTC-minute rows earlier",
    "seasonal_naive_weekly": "Predict avg_rate from 10,080 UTC-minute rows earlier",
}


def naive_forecasts(frame: pd.DataFrame, fit_mean: float) -> dict[str, np.ndarray]:
    """Predictions of all five baselines on `frame`; `fit_mean` must come from earlier data only."""
    preds = {
        "always_zero": np.zeros(len(frame)),
        "always_mean": np.full(len(frame), fit_mean),
    }
    for name, col in LAG_BASELINES.items():
        preds[name] = frame[col].to_numpy(dtype=np.float64)
    return preds
