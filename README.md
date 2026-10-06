# Residential Water-Consumption Forecasting

Forecasting one-minute whole-house water consumption of a single household
(AMPds2) with LightGBM and a linear benchmark, against naive baselines. The
evaluation is strictly chronological (train/validation/test, no shuffling).
The report is `tex/main.tex`.

## Data

The raw AMPds2 whole-house water file is included at `data/Water_WHW.csv`.

## Code layout

```
src/
  core/                 shared, importable functionality
    config.py           paths, lag/feature sets, final features, LightGBM settings
    data.py             load the feature table, chronological split
    features.py         lag and calendar features
    models.py           linear regression (OLS / median) and LightGBM fitting
    naive.py            the five naive baselines
    evaluation.py       MAE / RMSE / zero share, JSON summaries
  eda.py ... test_evaluation.py   one script per pipeline step (table below)
  plot_predictions.py   plots of saved predictions
tests/                  unit tests for src/core
run_all.py              runs all steps in order
```

Pipeline scripts import only from `src/core/`, never from each other.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| EDA and cleaning | `src/eda.py` | `output/eda/` (cleaned series, summary, figures) |
| Feature table | `src/build_features.py` | `output/build_features/` |
| Chronological split | `src/core/data.py` | used in memory by every later step |
| Naive baselines | `src/baselines.py` | `output/baselines/` |
| Model and loss comparison (linear regression, LightGBM; L2 and L1) | `src/compare_models.py` | `output/compare_models/` |
| Feature ablation | `src/feature_ablation.py` | `output/feature_ablation/` |
| Feature importance (grouped TreeSHAP) | `src/feature_importance.py` | `output/feature_importance/` |
| Residual autocorrelation | `src/residual_acf.py` | `output/residual_acf/` |
| Single test evaluation | `src/test_evaluation.py` | `output/test_evaluation/` |

## Running

From the repository root:

```sh
uv sync
uv run python run_all.py
uv run pytest
```

`run_all.py` runs every step in the order of the table above. Each step
writes its exact results to `output/<step>/summary.json`.

Prediction plots are made separately, from saved predictions (no refitting):
the validation split shows the baselines and the diagnostic models, the test
split the baselines and the final models.

```sh
uv run python src/plot_predictions.py --split valid --week 2
uv run python src/plot_predictions.py --split test --week 2 --day 2 --hours 16-22 --methods persistence lightgbm_l1
```

`--week` counts weeks from the start of the split, `--day` is 1 (Monday) to 7
(Sunday), `--hours` a local-time range. Method keys: `always_zero`,
`always_mean`, `persistence`, `seasonal_naive_daily`, `seasonal_naive_weekly`,
`linear_l2`, `linear_l1`, `lightgbm_l2`, `lightgbm_l1`. Plots go to
`output/prediction_plots/`.
