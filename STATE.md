# STATE.md

Snapshot of *where the project currently stands*. Changes often — every
time a stage finishes or an open decision resolves. Two things this file
deliberately does **not** contain:

- **Exact measured numbers** — those live in `output/`,
  produced by reproducible scripts (like eda, baseline or training scripts).
- **Settled judgment calls** (e.g. "this comparison is confounded, don't
  use it") — once something here resolves, its one-line conclusion moves
  to AGENTS.md's "Settled facts & scope decisions" and gets deleted from
  this file. If you're looking for something that isn't here anymore,
  check there.

Rules, protocol, and anything that shouldn't change session-to-session
live in AGENTS.md, not here.

Last updated: decided against the horizon ablation (coarsening the target
is not an informative comparison); Stage 4 is now a training-objective
ablation at 1-minute resolution (run; interpretation open).

## Where we are

EDA and feature derivation are complete: grid/integrity checks, meter
transition, target distribution, event structure, temporal profiles
(America/Vancouver local time), ACF (raw), timezone
verification (emitted as the `timezone` block in summary.json), and a
leakage-safe candidate feature set — all reproducible from
`src/initial_eda.py` → `output/eda_whw/summary.json`.

The feature table itself is now built: lags (1, 5, 15, 30, 60, 1440, 10080)
plus calendar features in both raw and cyclic encodings (local
time), warm-up NaN rows kept. Reproducible from `src/build_features.py`.

Stage 1 (chronological train/valid/test split) is built in `src/splits.py`:
Monday-00:00-local-aligned boundaries (~70/15/15), valid/test at full weekly
cycles, lag warm-up rows excluded from train. Downstream stages must obtain
splits via `split_series()` — never re-derive boundaries.

Stage 2 (naive/rule baselines) is complete: all five baselines were
evaluated on the chronological validation split using MAE, RMSE, and MASE.
Exact scores and split metadata are in `output/baselines/summary.json`.

Stage 3 (diagnostic models) is complete: linear regression and untuned
LightGBM were evaluated on the same validation split. LightGBM's MAE is
slightly below always-zero but remains above persistence; zero is the
optimal constant prediction, not necessarily the optimal feature-conditioned
rule. Results are in `output/diagnostic_models/summary.json`.

Stage 4 (training-objective ablation) is run on the validation split: OLS
vs. L1 linear regression and LightGBM with L2 / L1 / Huber / Poisson
objectives. Results (all metrics, zero-prediction share) are in
`output/objective_ablation/summary.json`; interpretation is still open. The
earlier horizon-aggregation ablation was dropped; its script and results are
archived in `archive/horizon_ablation/`. The final test split remains
untouched.

## Immediate next steps

1. Interpret the Stage 4 results across all metrics (not just MAE) and decide
   which objective(s) carry into Stage 5; record the conclusion in AGENTS.md.
2. Stage 5: feature-group ablation at 1-minute resolution with the chosen
   objective(s); then one evaluation on the test period, feature importance,
   residual ACF, light tuning.

## Modelling target

`y_t` is `avg_rate` (L/min) at native 1-minute resolution, predicted from
known calendar features and earlier lags only.

## Active feature set

Calendar features (hour, day-of-week, weekend; cyclic encodings for linear
regression, raw for LightGBM) and lags 1, 5, 15, 30, 60, 1440, 10080 from
`data/processed/whw_features_v1.parquet`. Still subject to the Stage 5
feature-group ablation.

Deliberately excluded: rolling statistics (AGENTS.md "Settled facts &
scope decisions") and weather (scope, per AGENTS.md Research Constraints).

## Evaluation metric

Report MAE, RMSE, and MASE against all explicit naive baselines. At each
resolution, MASE divides validation MAE by the training-only mean absolute
one-step target difference (persistence error). See
`AGENTS.md` → "Evaluation principles" for the settled rationale.
