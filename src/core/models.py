"""Model fitting: linear regression (OLS / median regression) and untuned LightGBM."""

import lightgbm as lgb
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from core.config import LGBM_OBJECTIVES, LGBM_PARAMS


def fit_linear(X_train, y_train, X_eval, loss: str) -> np.ndarray:
    """OLS (loss 'l2') or median regression (loss 'l1', QuantReg q=0.5); returns predictions on X_eval."""
    if loss == "l2":
        return LinearRegression().fit(X_train, y_train).predict(X_eval)
    if loss == "l1":
        fit = sm.QuantReg(y_train, sm.add_constant(X_train)).fit(q=0.5, max_iter=5000)
        return sm.add_constant(X_eval, has_constant="add") @ fit.params
    raise ValueError(f"unknown loss: {loss!r}")


def train_lgbm(X_train, y_train, loss: str) -> lgb.LGBMRegressor:
    """Untuned LightGBM (config.LGBM_PARAMS) with the objective for `loss`, fitted."""
    model = lgb.LGBMRegressor(objective=LGBM_OBJECTIVES[loss], **LGBM_PARAMS)
    return model.fit(X_train, y_train)


def fit_lgbm(X_train, y_train, X_eval, loss: str) -> np.ndarray:
    """Untuned LightGBM for `loss`; returns predictions on X_eval."""
    return train_lgbm(X_train, y_train, loss).predict(X_eval)
