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

Last updated: feature ablation run in two steps (calendar, then lags with
`hour` fixed) and the final feature set settled (`hour` + `lag_1`–`lag_15`);
`main.tex` updated to match; Stage 3 interpretation (L1 vs. L2) still open.

## Where we are

EDA and feature derivation are complete: grid/integrity checks, meter
transition, target distribution, event structure, temporal profiles
(America/Vancouver local time), ACF (raw), timezone
verification (emitted as the `timezone` block in summary.json), and a
leakage-safe candidate feature set — all reproducible from
`src/initial_eda.py` → `output/initial_eda/summary.json`.

The feature table itself is now built: lags (1–15, 30, 45, 60, 1440, 10080)
plus calendar features in both raw and cyclic encodings (local
time), warm-up NaN rows kept. Reproducible from `src/build_features.py`.

Stage 1 (chronological train/valid/test split) is built in `src/splits.py`:
Monday-00:00-local-aligned boundaries (~70/15/15), valid/test at full weekly
cycles, lag warm-up rows excluded from train. Downstream stages must obtain
splits via `split_series()` — never re-derive boundaries.

Stage 2 (naive/rule baselines) is complete: all five baselines were
evaluated on the chronological validation split using MAE and RMSE.
Exact scores and split metadata are in `output/baselines/summary.json`.

Stage 3 (diagnostic models, `src/diagnostic_models.py`) is run on the
validation split: linear regression and LightGBM, each under L2 and L1 (the
Huber/Poisson variants were dropped, see AGENTS.md). The L2 fits are the plain
diagnostic models. Results (all metrics, zero-prediction share) are in
`output/diagnostic_models/summary.json`; interpretation of the L1-vs-L2
trade-off is still open. The earlier
horizon-aggregation ablation was dropped; its script and results are archived
in `archive/horizon_ablation/`. The final test split remains untouched.

## Immediate next steps

1. Interpret the Stage 3 results across all metrics (not just MAE) and decide
   which loss(es) carry into the final model; record the conclusion in AGENTS.md.
2. Final modelling (lean protocol): small LightGBM grid (few hyperparameters)
   tuned train -> validation per loss; freeze features, loss(es) and
   hyperparameters; refit on train+valid (always-mean baseline on the same
   data); one test evaluation of the pre-declared models (final LightGBM,
   linear benchmark, all baselines).
3. On the final model: feature importance (grouped + single-feature
   permutation importance, TreeSHAP via `predict(pred_contrib=True)`) and
   residual ACF.

## Report

The final report is `main.tex` (single file). It was rewritten to the current
status after Stage 3/4 were reworked; results in it are validation-split only.
Update it at milestones or new results, but only after asking and checking in
with the user (AGENTS.md → "Living documentation"). Open `\todo{}` items mark
what is still unwritten.

## Modelling target

`y_t` is `avg_rate` (L/min) at native 1-minute resolution, predicted from
known calendar features and earlier lags only.

## Active feature set

Settled by the feature ablation (AGENTS.md): `hour` + `lag_1`–`lag_15`
(cyclic hour encoding for the linear benchmark), taken from
`output/build_features/whw_v100_with_features.parquet`, which keeps all
columns (the Stage 3 diagnostic models still use the full set).

Deliberately excluded: rolling statistics (AGENTS.md "Settled facts &
scope decisions") and weather (scope, per AGENTS.md Research Constraints).

## Evaluation metric

Report MAE and RMSE against all explicit naive baselines. See
`AGENTS.md` → "Evaluation principles" for the settled rationale.
