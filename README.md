# Residential Water-Consumption Forecasting

Reproducible time-series research using the AMPds2 whole-house water data.
The project uses chronological train/validation/test splits and keeps
generated data artifacts and measured results separate from source code.
See `AGENTS.md` for the research protocol and `STATE.md` for current status.

## Project map

Source files are currently kept directly under `src/`; the list below groups
them by their role in the workflow:

| File | Purpose |
| --- | --- |
| `src/initial_eda.py` | Clean the raw water series and produce EDA artifacts and figures. |
| `src/build_features.py` | Build the versioned 1-minute lag and calendar feature table. |
| `src/splits.py` | Shared chronological train/validation/test split. |
| `src/baselines.py` | Evaluate the Stage 2 naive baselines. |
| `src/diagnostic_models.py` | Evaluate Stage 3 linear regression and LightGBM models. |
| `src/horizon_ablation.py` | Compare 15-, 30-, and 60-minute targets for Stage 4. |
| `src/plot_val_predictions.py` | Plot validation actuals against saved Stage 2/3 predictions. |
| `src/ts_utils.py` | Shared time-series and serialization helpers. |

Tests are under `tests/`. Local input and processed data live under `data/`;
stage summaries and prediction artifacts live under `output/`.

## Reproducing the current stages

Run commands from the repository root. The raw AMPds2 file must be present at
`data/Water_WHW.csv`.

```sh
uv sync
uv run python src/initial_eda.py
uv run python src/build_features.py
uv run python src/baselines.py
uv run python src/diagnostic_models.py
uv run python src/horizon_ablation.py
uv run python src/plot_val_predictions.py
```

The baseline and diagnostic-model scripts write both `summary.json` and
`validation_predictions.parquet` in their respective `output/` folders.
The plotter reads those saved predictions; it does not refit models. Edit
`VALIDATION_WEEK`, `DAY_OF_WEEK`, or `PREDICTIONS_TO_PLOT` near the top of
`src/plot_val_predictions.py` to choose the displayed local-calendar period
and methods. Week numbers run from 1 to 13; weekdays are numbered Monday=1
through Sunday=7. Set `DAY_OF_WEEK = None` to show the whole week.
Method keys are `always_zero`, `always_mean`, `persistence`,
`seasonal_naive_daily`, `seasonal_naive_weekly`, `linear_regression`, and
`lightgbm`.

Plots are written under `output/validation_predictions/figs/`.
