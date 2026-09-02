# AGENTS.md

## Project

University ML/time-series project on **residential water-consumption prediction** using the **AMPds2 (Almanac of Minutely Power dataset v2)**.

Main research questions:

1. How well can water consumption be predicted with one or two ML models?
2. Which features are most important?
3. How strong is temporal autocorrelation?
4. How can its effect be handled/exploited without temporal leakage?

The final work is an academic report supported by reproducible code and experiments.

## Research Constraints

These rules are important:

- Treat the problem as a time-series forecasting problem.
- Never randomly shuffle the time series for the main evaluation.
- Train/validation/test splits must respect chronological order.
- Never allow future observations to influence past predictions.
- Distinguish clearly between hypotheses and measured results.
- Do not fabricate or assume experimental results.
- Do not add features or models solely because they improve a metric; consider their methodological justification.
- Keep the project focused on the research questions above.


## Dataset

AMPds2 contains **730 days of 1-minute measurements** (1,051,200 rows).

Relevant files:

- `data/Water_WHW.csv` — whole-house water (primary target candidate)

Water CSV columns:

- `unix_ts` — Unix timestamp
- `counter` — cumulative volume, in Liters L
- `avg_rate` — volume consumed during the sampling interval; for 1-minute data, effectively L/min
- `inst_rate` — instantaneous flow-rate estimate

Water meters work through **pulses**: each pulse represents a fixed volume passing through the meter. The old meters used roughly 1 gallon (3.785 L) per pulse; the replacement V100 meters use roughly 0.5 L per pulse.

There was a meter replacement on **2012-07-14 (Unix timestamp 1342287780)**. This changes measurement resolution and must be considered in preprocessing/sensitivity analysis.

Initial pandas inspection found **no missing values** in the water CSV.

### Water-meter period

For the primary analysis, use only observations from the documented V100
meter transition onward (Unix timestamp 1342287780). Earlier observations
come from the previous meter configuration and are to be excluded to avoid mixing
different measurement resolutions.

The cumulative `counter` can potentially be retained for validation, but `avg_rate` is the
primary interval-level modelling target.


## Current modelling hypothesis

Primary target candidate:

`Water_WHW.csv["avg_rate"]`

For one-step-ahead forecasting:

`y_t = avg_rate_t`

using only information available at or before `t-1`.

The exact forecasting horizon is still to be finalized.

## Temporal autocorrelation

Temporal autocorrelation means observations close together in time are statistically dependent.

Important candidate lags for 1-minute data:

- 1 = 1 minute
- 5 = 5 minutes
- 15 = 15 minutes
- 30 = 30 minutes
- 60 = 1 hour
- 1440 = 1 day
- 10080 = 1 week

Next major analysis: calculate and visualize the **ACF** of WHW consumption and use it to motivate lag selection.

Distinguish:

- **Exploiting autocorrelation:** adding lagged consumption as features.
- **Mitigating autocorrelation/leakage:** using chronological splits and ensuring future observations cannot enter preprocessing/training.

Residual ACF should later be checked to see whether substantial temporal structure remains unexplained.

## Planned experiments

Use chronological train/validation/test splits — **never random splitting** for the main forecasting experiment.

Start with simple baselines:

1. Persistence: `y_hat_t = y_{t-1}`
2. Previous-day: `y_hat_t = y_{t-1440}`

Then progressively add feature groups:

1. Calendar/time features
2. Calendar + recent consumption lags
3. Calendar + recent + daily/weekly lags
4. Optional full selected feature set

This ablation is important because it directly tests how much predictive performance comes from temporal dependence.

Candidate models:

- tree-based regression model, LightGBM as a start
- simple linear regression benchmark needed?

Do not add many models without justification. Interpretability and the temporal-autocorrelation question are more important than model count.

## Candidate features

Calendar:

- hour
- day of week
- weekend
- month/day of year
- preferably cyclical encodings for periodic variables

Consumption lags:

- 1, 5, 15, 30, 60, 1440, 10080

## Immediate next steps

Done so far (results documented in REPORT.md):

1. Load WHW and timestamps; verified a perfect 1-minute grid, no missing values.
2. Investigated the 2012 meter-resolution transition; data limited to the V100
   period (`scripts/eda_whw.py` → `data/processed/whw_v100.parquet`).
3. Explored WHW distribution, zero-rate proportion and time patterns.
4. Calculated ACF; candidate lag set motivated.
5. Autocorrelation/feature walkthrough notebook:
   `notebooks/autocorrelation_and_features.py`.

Next:

6. Finalize the exact prediction horizon (working assumption: one-step-ahead, h=1).
7. Build chronological train/validation/test splits.
8. Implement persistence and previous-day baselines.
9. Implement first simple ML model (LightGBM as a start).
10. Run feature-ablation experiments.
11. Analyze feature importance and residual ACF.
12. Tune only a small number of meaningful hyperparameters.
13. Document results reproducibly.

## Living documentation

Keep the documents in sync with the actual project state — they are the source
of truth for any future session (human or agent):

- Whenever a **decision** is made (e.g., forecasting horizon, split sizes,
  final feature set, scope changes), update AGENTS.md immediately.
- Whenever a **measured result** is produced (EDA findings, baseline scores,
  ablation outcomes), add it to REPORT.md with its exact numbers.
- Never let the documents contradict the code or the data; if numbers change,
  update or regenerate them instead of keeping stale values.
- Preserve the distinction between measured results and hypotheses in both
  documents.
- Features/models/scripts that are dropped or replaced must be removed from
  the documents as well, so scope and documentation stay aligned.

## Coding principles

- Use Python/pandas/numpy/scikit-learn unless another library is justified.
- Keep exploratory notebooks separate from reusable source code.
- Avoid data leakage at every stage.
- Never randomly shuffle the time series for the main evaluation.
- Make preprocessing reproducible.
- Prefer clear, small functions over one large notebook.
- Save plots/results reproducibly rather than manually editing them.
- Do not fabricate results; clearly label hypotheses vs measured results.
- Keep the project scope focused on the research questions above.

