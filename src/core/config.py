"""Paths, feature sets and model settings shared by all pipeline scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output"
RAW_CSV = REPO_ROOT / "data" / "Water_WHW.csv"
V100_PARQUET = OUTPUT_DIR / "eda" / "whw_v100.parquet"
FEATURES_PARQUET = OUTPUT_DIR / "build_features" / "whw_v100_with_features.parquet"

LOCAL_TZ = "America/Vancouver"
TARGET = "avg_rate"

# Candidate lags (1-minute units): dense 1-15 (ACF is highest there), then
# hour-scale, daily, weekly. Shared by the ACF analyses, feature table and models.
LAGS = [*range(1, 16), 30, 45, 60, 1440, 10080]
LAG_COLS = [f"lag_{k}" for k in LAGS]
IMMEDIATE = [f"lag_{k}" for k in range(1, 16)]
HOUR_SCALE = ["lag_30", "lag_45", "lag_60"]
SEASONAL = ["lag_1440", "lag_10080"]

# Feature sets of the diagnostic models: calendar + all lags; cyclic calendar for the linear model.
FEATURES_LINEAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend"] + LAG_COLS
FEATURES_TREE = ["hour", "dow_local", "weekend"] + LAG_COLS

# Final feature set, chosen by the feature ablation; used by every final model.
FINAL_FEATURES = ["hour"] + IMMEDIATE

# Untuned by design: fixed values and seed, no search.
LGBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "random_state": 42,
    "verbosity": -1,
}
# Training losses compared for every model class, and their LightGBM objectives.
LOSSES = ("l2", "l1")
LGBM_OBJECTIVES = {
    "l2": "regression_l2",
    "l1": "regression_l1",
}
