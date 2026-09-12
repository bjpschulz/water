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
- `counter` — cumulative volume, in Liters (L)
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
primary interval-level modelling quantity, at whatever resolution is chosen (see Forecasting Horizon Decision below).

## Forecasting Horizon Decision

**Status: decided.**

### The problem this decision addresses

The raw 1-minute `avg_rate` series is heavily zero-inflated: most one-minute
intervals have zero consumption, with occasional short bursts. A one-step-ahead
model on raw 1-minute data (`y_t = avg_rate_t`, predicted from information at
`t-1` or earlier) risks trivially predicting near-zero everywhere and scoring
deceptively well on standard error metrics without capturing meaningful structure.
This directly affects RQ1 (predictability), RQ2 (feature importance — signal is
hard to see under sparsity), and RQ3 (autocorrelation — ACF on a mostly-zero
series is harder to interpret).

### Options considered

**Option A — keep 1-minute resolution, reframe as two-part (hurdle) model:**
predict (i) probability of any flow in the next minute, and (ii) magnitude given
flow.
- Pro: preserves full native granularity; realistic for real-time/operational use cases (e.g. leak detection).
- Con: requires two-part modelling machinery; standard regression metrics (RMSE/MAE) become misleading and would need to be replaced or supplemented, adding scope beyond what's needed to answer the stated RQs.

**Option B — aggregate to a coarser fixed time block, predict total consumption in the next block:**
- Pro: smooths zero-inflation, plain regression metrics behave sensibly, calendar/lag features show cleaner signal, ACF is more interpretable, aligns with standard practice in load/consumption forecasting literature.
- Con: loses minute-level granularity; the aggregation window itself is a choice requiring justification; degree of zero-inflation improvement must be verified empirically, not assumed.

**Option C — keep 1-minute resolution but predict a single value further ahead (e.g. `t+15`) instead of the next block's total:**
- Rejected: does not address zero-inflation (still predicting a single sparse minute value, merely shifted in time); weaker fit for the stated RQs than B.

### Decision

**Option B is adopted.** Aggregation window: **15-minute blocks** (working choice;
may be revisited to 60-minute if 15-minute blocks still show substantial
zero-inflation during EDA — this must be checked empirically and logged in
REPORT.md, not assumed).

New target definition:

y_t = sum(avg_rate) over the 15-minute block ending at t

predicted using only information available at or before the end of the
*previous* 15-minute block (`t-1` in block-indexed time, i.e. strictly no
information from within the block being predicted).

Forecasting horizon: **one block ahead (h=1 in block units)**, i.e. predict the
next 15-minute total from current and past block(s).

## Current modelling hypothesis

Primary target: 15-minute aggregated `avg_rate` (total liters per 15-minute
block), derived from `Water_WHW.csv`, V100 period only.

For one-step-ahead forecasting at the new resolution:

`y_t = sum(avg_rate)` for the 15-minute block ending at `t`

using only information available at or before the end of block `t-1`.

## Temporal autocorrelation

Temporal autocorrelation means observations close together in time are statistically dependent.

**Note:** candidate lags below are expressed in 15-minute blocks, not minutes,
following the horizon decision above. 1 block = 15 minutes.

Important candidate lags for 15-minute aggregated data:

- 1 = 15 minutes
- 4 = 1 hour
- 96 = 1 day
- 672 = 1 week

These replace the original 1-minute-based lag set (1, 5, 15, 30, 60, 1440,
10080 minutes), which was derived for the raw 1-minute series and is no longer
the operative resolution.

Next major analysis: recompute the ACF on the resampled 15-minute series (the
existing 1-minute ACF work is a hypothesis-generating precursor, not the final
analysis) and use it to confirm or adjust the lag set above.

Distinguish:

- **Exploiting autocorrelation:** adding lagged consumption (at the block
  resolution) as features.
- **Mitigating autocorrelation/leakage:** using chronological splits and
  ensuring future blocks cannot enter preprocessing/training.

Calendar features (hour, day-of-week, etc.) are a related but distinct
mechanism: they encode known-in-advance seasonal structure without relying on
the target's own past values, and can partly substitute for daily/weekly lag
features. The ablation below is designed to test this directly.

Residual ACF should later be checked to see whether substantial temporal
structure remains unexplained after modelling.

## Planned experiments

Use chronological train/validation/test splits — **never random splitting**
for the main forecasting experiment. Test set should be a contiguous final
block of the series, sized to cover full weekly cycles given the strong
daily/weekly seasonality expected in this data.

Start with simple baselines, now at 15-minute block resolution:

1. Persistence: `y_hat_t = y_{t-1}` (previous 15-minute block)
2. Previous-day: `y_hat_t = y_{t-96}` (same block, previous day)
3. Previous-week: `y_hat_t = y_{t-672}` (same block, previous week) — new
   addition, since weekly seasonality is now cheap to test as a baseline at
   this resolution.

Then progressively add feature groups:

1. Calendar/time features
2. Calendar + recent consumption lags (1, 4 blocks)
3. Calendar + recent + daily/weekly lags (96, 672 blocks)
4. Optional full selected feature set

This ablation is important because it directly tests how much predictive
performance comes from temporal dependence, and — via the calendar-only vs.
lag-only comparison — whether daily/weekly autocorrelation is better explained
by calendar structure or by the target's own recent history.

Candidate models:

- tree-based regression model, LightGBM as a start
- simple linear regression benchmark needed?

Do not add many models without justification. Interpretability and the
temporal-autocorrelation question are more important than model count.

## Candidate features

Calendar:

- hour
- day of week
- weekend
- month/day of year
- preferably cyclical encodings for periodic variables

Consumption lags (in 15-minute blocks):

- 1, 4, 96, 672

## Evaluation (open item)

Not yet finalized — needs a decision before baselines are run:

- Primary metric candidate: MAE or RMSE on the 15-minute total, benchmarked
  against persistence/previous-day/previous-week baselines (e.g. via a
  MASE-style ratio).
- Zero-inflation at 15-minute resolution must be checked empirically during
  EDA before assuming standard regression metrics are adequate; if a large
  fraction of 15-minute blocks are still zero, this should be revisited.

## Immediate next steps

Done so far (results documented in REPORT.md, at 1-minute resolution):

1. Load WHW and timestamps; verified a perfect 1-minute grid, no missing values.
2. Investigated the 2012 meter-resolution transition; data limited to the V100
   period (`scripts/eda_whw.py` → `data/processed/whw_v100.parquet`).
3. Explored WHW distribution, zero-rate proportion and time patterns at
   1-minute resolution.
4. Calculated ACF at 1-minute resolution; motivated the original candidate
   lag set (now superseded by the horizon decision above).
5. Autocorrelation/feature walkthrough notebook (1-minute resolution):
   `notebooks/autocorrelation_and_features.py`.

Next (reflecting the Option B decision):

6. Resample the V100-period 1-minute series into 15-minute blocks (sum
   `avg_rate` per block; cross-check against `counter` differences).
7. Re-run EDA on the resampled series: check zero-inflation rate at this
   resolution (do not assume it improves — measure it).
8. Recompute ACF on the 15-minute series; confirm or adjust the block-lag
   candidates above.
9. Finalize the evaluation metric(s) given the resampled series' properties.
10. Build chronological train/validation/test splits on the resampled series.
11. Implement persistence, previous-day, and previous-week baselines.
12. Implement first simple ML model (LightGBM as a start).
13. Run feature-ablation experiments (calendar / +recent lags / +daily-weekly lags / full).
14. Analyze feature importance and residual ACF.
15. Tune only a small number of meaningful hyperparameters.
16. Document results reproducibly.

## Living documentation

Keep the documents in sync with the actual project state — they are the source
of truth for any future session (human or agent):

- Whenever a **decision** is made (e.g., forecasting horizon, split sizes,
  final feature set, scope changes), update AGENTS.md immediately, including
  the reasoning and rejected alternatives where relevant (see Forecasting
  Horizon Decision above as the template for this).
- Whenever a **measured result** is produced (EDA findings, baseline scores,
  ablation outcomes), add it to REPORT.md with its exact numbers.
- Never let the documents contradict the code or the data; if numbers change,
  update or regenerate them instead of keeping stale values.
- Preserve the distinction between measured results and hypotheses in both
  documents.
- Features/models/scripts that are dropped or replaced must be removed from
  the documents as well, so scope and documentation stay aligned. (The
  1-minute-resolution lag set and 1-minute ACF work are retained above only
  as historical/superseded context, clearly marked as such — not as active
  scope.)

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