# STATE.md

Snapshot of *where the project currently stands*. Changes often — every
time a stage finishes or an open decision resolves. Two things this file
deliberately does **not** contain:

- **Exact measured numbers** — those live in `results/`,
  produced by reproducible scripts (like eda, baseline or training scripts).
- **Settled judgment calls** (e.g. "this comparison is confounded, don't
  use it") — once something here resolves, its one-line conclusion moves
  to AGENTS.md's "Settled facts & scope decisions" and gets deleted from
  this file. If you're looking for something that isn't here anymore,
  check there.

Rules, protocol, and anything that shouldn't change session-to-session
live in AGENTS.md, not here.

Last updated: after completing the Stage 4 horizon ablation and selecting
the 60-minute modeling target (`src/horizon_ablation.py`;
`results/horizon_ablation/summary.json`).

## Where we are

EDA and feature derivation are complete: grid/integrity checks, meter
transition, target distribution, event structure, temporal profiles
(America/Vancouver local time), ACF (raw), timezone
verification (emitted as the `timezone` block in summary.json), and a
leakage-safe candidate feature set — all reproducible from
`src/initial_eda.py` → `results/eda_whw/summary.json`.

The feature table itself is now built: lags (1, 5, 15, 30, 60, 1440, 10080)
plus calendar features in both raw and cyclic encodings (local
time), warm-up NaN rows kept. Reproducible from `src/build_features.py`.

Stage 1 (chronological train/valid/test split) is built in `src/splits.py`:
Monday-00:00-local-aligned boundaries (~70/15/15), valid/test at full weekly
cycles, lag warm-up rows excluded from train. Downstream stages must obtain
splits via `split_series()` — never re-derive boundaries.

Stage 2 (naive/rule baselines) is complete: all five baselines were
evaluated on the chronological validation split using MAE, RMSE, and MASE.
Exact scores and split metadata are in `results/baselines/summary.json`.

Stage 3 (diagnostic models) is complete: linear regression and untuned
LightGBM were evaluated on the same validation split. LightGBM's MAE is
slightly below always-zero but remains above persistence; zero is the
optimal constant prediction, not necessarily the optimal feature-conditioned
rule. Results are in `results/diagnostic_models/summary.json`.

Stage 4 (horizon aggregation ablation) is complete for 15-, 30-, and
60-minute block-volume targets. Metrics, training target zero proportions,
split metadata, and feature lists are in
`results/horizon_ablation/summary.json`. Validation evidence selects
60-minute blocks for Stage 5: LightGBM beats the naive baselines at that
horizon and has the lowest MAE/RMSE per minute among the tested models and
baselines. The final test split remains untouched.

## Immediate next steps

Next:
1. Run the Stage 5 feature-group ablation at the selected 60-minute
   resolution, comparing calendar-only, lag-only, calendar + one-block lag,
   and calendar + one-block + daily/weekly lags for linear regression and
   LightGBM.
2. Select features based on the validation ablation, then evaluate the
   selected setup once on the untouched test period.
3. Analyze feature importance and residual ACF; tune only a small number
   of meaningful hyperparameters if justified.
4. Save all measured results reproducibly under `results/`.

## Forecasting horizon / resolution

**Status: selected from Stage 4 validation evidence.** Use 60-minute
non-overlapping block-volume targets for Stage 5. This is a validation-time
resolution choice; do not use the final test period to revisit it.

## Current modelling target

For each fixed UTC-aligned 60-minute block, `y_t` is the sum of its 1-minute
`avg_rate` values (liters consumed per block). Predict at block start using
only completed earlier blocks and known calendar features. The 1-minute
series remains the resolution for the separate autocorrelation analysis.

## Active feature set

Stage 5 candidates at the selected 60-minute resolution, still subject to
the feature-group ablation:

1. Calendar features from block start in America/Vancouver: hour and
   day-of-week, with cyclic encodings for linear regression and raw values
   for LightGBM (`FEATURES.md`).
2. Completed-block consumption lags: `lag_1_block` (recent),
   `lag_24_block` (daily), and `lag_168_block` (weekly), derived without
   looking into the target block (`src/horizon_ablation.py`).

Deliberately excluded: rolling statistics (AGENTS.md "Settled facts &
scope decisions") and weather (scope, per AGENTS.md Research Constraints —
never part of the candidate set).

The chronological Stage 4 split excludes the one-week lag warm-up from
training; validation and test feature rows contain no lag NaNs.

## Evaluation metric

Report MAE, RMSE, and MASE against all explicit naive baselines. At each
resolution, MASE divides validation MAE by the training-only mean absolute
one-step target difference (persistence error). The Stage 4 comparison also
reports MAE/RMSE per minute to compare block errors in common units. See
`AGENTS.md` → "Evaluation principles" for the settled rationale.
