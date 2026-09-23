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

The narrative write-up (why these choices were made, what they mean) is
the user's own report, written separately and not tracked in either of
these files.

Last updated: after Phase 4 equivalent work (autocorrelation/feature
derivation), before Stage 1 (chronological splits).

## Where we are

EDA and feature derivation are complete: grid/integrity checks, meter
transition, target distribution, event structure, temporal profiles, ACF
(raw + occurrence-indicator), timezone verification, and a leakage-safe
candidate feature set — all reproducible from `scripts/eda_whw.py` →
`results/eda_whw/summary.json`.

Stage 1 (chronological train/val/test split) has **not** been built yet.
No model has been fit. No baseline has been run.

## Immediate next steps

Next:
1. Build chronological train/validation/test splits on the V100-period
   1-minute series (Stage 1). Test set sized to cover full weekly cycles.
2. Compute naive/rule baselines at 1-minute resolution on the validation
   set (Stage 2): always-zero, always-mean, persistence, seasonal-naive-daily,
   seasonal-naive-weekly. Save MAE/RMSE reproducibly under `results/`.
3. Fit the diagnostic linear regression and quick LightGBM (Stage 3) using
   calendar + full lag set. Same validation set, same treatment.
4. Decision point (Stage 4): compare Stage 3 vs Stage 2. Update "Open
   decision" below to "decided" with the actual numbers, then migrate it
   to AGENTS.md's settled-facts list.
5. Depending on step 4: proceed with full modeling at 1-minute resolution,
   or revisit Option A/B below with measured justification.
6. Run the feature-group ablation (calendar / +recent lags /
   +daily-weekly lags / full) at the settled resolution.
7. Analyze feature importance and residual ACF.
8. Tune a small number of meaningful hyperparameters.
9. Document results reproducibly under `results/`.

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

In priority order, each justified by a measurement in `results/eda_whw/summary.json`:

1. Recent consumption lags: `lag_1, lag_5, lag_15, lag_30, lag_60`
   (ACF decay region + event durations)
2. Daily/weekly lags: `lag_1440, lag_10080` (24h/168h ACF bumps)
3. Occurrence lag: `occ_lag_1` (indicator ACF decays slower than raw ACF
   in the 15–60 min range — motivates tracking occurrence separately)
4. Calendar: `hour_sin, hour_cos` (diurnal profile, local time),
   `dow_local, weekend` (cheap; expected weak relative to hour-of-day)

Deliberately excluded: rolling statistics, weather — see AGENTS.md
"Settled facts & scope decisions" for why.

**Known issue for the modeling pipeline:** `lag_10080` produces a 7-day
warm-up NaN tail (10,080 rows). The chronological split must place that
tail inside the training region, never at a split boundary.

## Evaluation metric — open item

Not finalized. Leading candidate: MAE and/or RMSE benchmarked against
persistence/seasonal-naive baselines via a MASE-style ratio. Open
sub-question: whether the MASE scaling denominator uses in-sample
seasonal-naive error (Hyndman's original definition) or the same held-out
set (simpler, less standard). Settle this once Stage 2–4 numbers exist —
don't lock in a metric before there's data to check it against. See
AGENTS.md → "Evaluation principles" for the stable reasoning behind why
both MAE and RMSE are tracked regardless of which becomes primary.