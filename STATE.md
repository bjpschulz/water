# STATE.md

Snapshot of *where the project currently stands*. Changes often — every
time a stage finishes or an open decision resolves. Two things this file
deliberately does **not** contain:

- **Exact measured numbers** — those live in `results/*/summary.json`,
  produced by reproducible scripts. Cite the path, don't retype the value.
- **Settled judgment calls** (e.g. "this comparison is confounded, don't
  use it") — once something here resolves, its one-line conclusion moves
  to AGENTS.md's "Settled facts & scope decisions" and gets deleted from
  this file. If you're looking for something that isn't here anymore,
  check there.

Rules, protocol, and anything that shouldn't change session-to-session
live in AGENTS.md, not here.

Last updated: after building the Stage 1 chronological splits
(`src/splits.py`, consumed in-memory via `split_series()`), before Stage 2
(naive baselines).

## Where we are

EDA and feature derivation are complete: grid/integrity checks, meter
transition, target distribution, event structure, temporal profiles
(America/Vancouver local time), ACF (raw + occurrence-indicator), timezone
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

No model has been fit. No baseline has been run.

## Immediate next steps

Next:
1. Compute naive/rule baselines at 1-minute resolution on the validation
   set (Stage 2), in a new `src/baselines.py` importing `load_features` /
   `split_series` from `src/splits.py`: always-zero, always-mean (train
   mean), persistence, seasonal-naive-daily, seasonal-naive-weekly (fixed
   UTC-minute lag convention, per AGENTS.md Stage 2). Baselines needing
   history before valid's first row (persistence, seasonal-naive) get it by
   concatenating train+valid — contiguous by construction. Save MAE/RMSE
   reproducibly under `results/`.
2. Fit the diagnostic linear regression and quick LightGBM (Stage 3) using
   calendar + full lag set. Same validation set, same treatment. Train on
   the train split as-is (warm-up NaN rows already excluded by Stage 1).
   Note: `lightgbm` is not yet a dependency — add it to `pyproject.toml`
   when this step starts.
3. Decision point (Stage 4): compare Stage 3 vs Stage 2. Update "Open
   decision" below to "decided" with the actual numbers, then migrate it
   to AGENTS.md's settled-facts list.
4. Depending on step 3: proceed with full modeling at 1-minute resolution,
   or revisit Option A/B below with measured justification.
5. Run the feature-group ablation (calendar / +recent lags /
   +daily-weekly lags / full) at the settled resolution.
6. Analyze feature importance and residual ACF.
7. Tune a small number of meaningful hyperparameters.
8. Document results reproducibly under `results/`.

## Open decision: forecasting horizon / resolution

Not an action item itself — it resolves as a byproduct of steps 1–4 above
(specifically Stage 4).

**Status: pending empirical validation (Stage 4, not yet run).**
Working default in the meantime: native 1-minute resolution.

This is gated by AGENTS.md's evidence-gating rule — it does not become
"decided" until Stage 2–4 numbers exist.

Context relevant to judging Stage 4, once it runs:
- Median 1-minute `avg_rate` is 0 (`target_stats.quantiles.p50` in
  `results/eda_whw/summary.json`). Always-predicting-0 is already the
  MAE-optimal constant rule at this resolution, so any MAE improvement
  over it from Stage 3 is real signal, not exploited sparsity.
- Calendar features and same-position lags are complementary, not
  redundant (calendar = expected value at a temporal position, lag =
  realized value) — relevant when interpreting Stage 3/5 feature
  importances, not itself a resolution argument.

Fallback if Stage 4 finds no real headroom at 1-minute resolution:
- **Option A** — hurdle/two-part model at native resolution (probability
  of flow, then magnitude given flow). Needs different metrics than plain
  RMSE/MAE.
- **Option B** — aggregate to a fixed block (e.g. 15 min), predict block
  totals. A rough estimate from `events.mean_events_per_day` (78.3/day,
  `summary.json`) suggests ~35–55% zero at 15 min — **unverified**, must
  be measured directly before this option is adopted, not assumed.

When this decision resolves: update the status line above, move a
one-line settled version into AGENTS.md, and delete this section from
STATE.md.

## Current modelling target (working hypothesis, pending the decision above)

`y_t = avg_rate_t`, native 1-minute, V100 period only, predicted using
only information available at or before `t-1`.

If the horizon decision moves to Option B: reformulate to
`y_t = sum(avg_rate)` over the 15-minute block ending at `t`, predicted
from information available at or before the end of block `t-1`. Do not
run both formulations as if both were simultaneously current — replace
this section's content, don't append to it.

## Active feature set

Current, still subject to change by the Stage 5+ ablation. Every feature
uses only information from `t-1` or earlier (leakage-safe shift semantics).
Items 1, 2, and 4 are built into `data/processed/whw_features_v1.parquet`
(`src/build_features.py`); item 3 is not built yet. Encoding rationale per
model family: `FEATURES.md`.

In priority order, each justified by a measurement in `results/eda_whw/summary.json`:

1. Recent consumption lags: `lag_1, lag_5, lag_15, lag_30, lag_60`
   (ACF decay region + event durations) — built
2. Daily/weekly lags: `lag_1440, lag_10080` (24h/168h ACF bumps) — built
3. Occurrence lag: `occ_lag_1` (indicator ACF decays slower than raw ACF
   in the 15–60 min range — motivates tracking occurrence separately) —
   deferred, not yet in the feature table
4. Calendar: `hour, hour_sin, hour_cos` (diurnal profile, local time),
   `dow_local, dow_sin, dow_cos, weekend` (cheap; expected weak relative
   to hour-of-day) — built; raw ints for tree models, sin/cos for the
   linear benchmark (FEATURES.md)

Deliberately excluded: rolling statistics (AGENTS.md "Settled facts &
scope decisions") and weather (scope, per AGENTS.md Research Constraints —
never part of the candidate set).

The former "known issue" about the `lag_10080` warm-up NaN tail is
resolved: the Stage 1 split (`src/splits.py`) excludes it from train, so no
split contains NaN lag features.

## Evaluation metric — open item

Not finalized. Leading candidate: MAE and/or RMSE benchmarked against
persistence/seasonal-naive baselines via a MASE-style ratio. Open
sub-question: whether the MASE scaling denominator uses in-sample
seasonal-naive error (Hyndman's original definition) or the same held-out
set (simpler, less standard). Settle this once Stage 2–4 numbers exist —
don't lock in a metric before there's data to check it against. See
AGENTS.md → "Evaluation principles" for the stable reasoning behind why
both MAE and RMSE are tracked regardless of which becomes primary.