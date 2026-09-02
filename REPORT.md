# Residential Water Consumption Prediction — Report Notes (Draft)

Status: skeleton of bullet-point findings to be expanded into the final academic
report. The document is organized **chronologically as the phases the project
went through**: what was found during personal exploration, what the
systematic EDA then established or corrected, which scope decisions were made,
and where that leaves the modeling plan.

All numbers are **measured** on the data unless explicitly labeled
*(interpretation)* or *(hypothesis)*. Reproducible sources are listed in the
appendix.

---

## The data (context)

- Source: **AMPds2** (Almanac of Minutely Power dataset v2), 1-minute resolution
  measurements from a single occupied residential house (Burnaby, BC, Canada).
- Coverage: **730 days**, 2012-04-01 07:00 UTC to 2014-04-01 06:59 UTC
  (= local midnight to local midnight, Pacific time); exactly **1,051,200 rows**.
- File used: `data/Water_WHW.csv` (whole-house water), columns:
  - `unix_ts` — Unix timestamp (established later to be true Unix/UTC time,
    see Phase 3)
  - `counter` — cumulative volume in liters
  - `avg_rate` — volume consumed during the 1-minute interval (L/min) — **target**
  - `inst_rate` — instantaneous flow-rate estimate (secondary, not modeled)
- Measurement principle: the meter emits **pulses**, each representing a fixed
  volume. Old meter: ~1 gallon (3.785 L) per pulse. Replacement V100 meter:
  ~0.5 L per pulse.

## Phase 1 — Personal first exploration

Goal: get familiar with the raw file before building any tooling. Done
interactively in `notebooks/explore_whw.py`; findings below were established
by hand and are **provisional** — the systematic EDA in Phase 2 was explicitly
designed to confirm or correct them.

### What was inspected and observed

- Loaded the CSV, converted `unix_ts` to datetime, sorted chronologically.
- Shape and coverage: 1,051,200 rows, 2012-04-01 to 2014-04-01; row count
  matches the expected 730 × 24 × 60.
- First data-quality look: no missing values; most common interval is 1 minute;
  no duplicate timestamps. (No systematic gap audit yet.)
- Target distribution: large zero share; heavy right skew (histograms of all
  values and of non-zero values).
- Time-series overview and a one-week example: consumption is bursty and
  event-like rather than smooth.
- First temporal profiles: clear hour-of-day structure; day-of-week profile
  looks nearly flat.
- Meter transition (known from AMPds2 documentation, 2012-07-14): counted rows
  on both sides, inspected the raw rows around the switch, and compared mean
  `avg_rate` (0.272 old vs 0.564 V100) and zero proportions (93.7% vs 85.5%)
  between the two meter periods.
- **Decision taken here (later verified in Phase 2):** restrict all analysis to
  the V100 meter period (`unix_ts >= 1342287780`) to avoid mixing two
  measurement resolutions.

### What this phase could not establish

- Whether the 1-minute grid is truly gap-free (only spot-checked).
- Whether `avg_rate` is internally consistent with `counter`.
- The exact pulse resolution of each meter (and whether the documented
  transition timestamp is clean).
- Any quantification of autocorrelation (ACF).
- The timezone semantics of `unix_ts` (UTC vs local) — profiles were read in
  the naive timestamp frame.
- No reproducible artifacts (numbers/figures lived only in the session).

## Phase 2 — Systematic, reproducible EDA

Goal: turn the provisional observations into verified facts and produce citable
artifacts. Implemented once in `scripts/eda_whw.py`; rerunning it regenerates
`data/processed/whw_v100.parquet`, `results/eda_whw/summary.json` and all
figures. Every number below is reproducible.

### Data-quality validation (confirms and extends Phase 1)

- Confirmed: no missing values, strictly monotone timestamps, zero duplicates.
- **New — perfect 1-minute grid**: zero gaps and zero off-grid rows in both the
  full record and the V100 subset. *(Prerequisite for lag indices like
  1440 = exactly 1 day.)*
- **New — no negative values**: no negative `avg_rate`, no counter decreases.
- **New — internal consistency**: `avg_rate_t = counter_t - counter_{t-1}`
  holds to within 1.1e-11 L (max absolute deviation). `counter` is therefore
  redundant for modeling and is retained only for validation.

### Meter replacement, verified (turns the Phase-1 decision into checked fact)

- V100 transition at **unix_ts 1342287780** = 2012-07-14 17:43 UTC
  (~10:43 local PDT). Excluded: 150,403 rows. Kept: **900,797 rows =
  625.55 days**.
- **Pulse-size verification from counter increments** (the physical mechanism
  behind the resolution change):
  - Old meter: smallest positive increment 3.785 L; most common increments are
    exact multiples (3.785: 8,441x; 7.57: 867x; 11.355: 193x).
  - V100: most common increments are multiples of 0.5 L
    (0.5: 26,602x; 1.0: 15,465x; 7.0: 13,625x).
  - Single anomaly: one 0.053 L increment at 2012-07-15 00:02 UTC, ~6 h after
    the swap — a one-off counter settling onto the 0.5 L grid. Negligible.
- Counter is **continuous across the swap** (86,210.947 L on both sides) — no
  reset, so `counter` remains usable for validation.
- **Correction to the Phase-1 comparison** *(interpretation)*: the old-vs-V100
  difference in mean and zero proportion is **confounded by seasonality** (the
  two periods cover different parts of the year; the old meter conserves volume
  but clumps it into spikes). It must not be read as a meter or behavioral
  effect.
- Output: `data/processed/whw_v100.parquet` (datetime index; `unix_ts`,
  `counter`, `avg_rate`, `inst_rate` columns).

### Target distribution (V100 period) — first systematic measurement

- Mean 0.564 L/min, std 1.937, max 35 L/min — heavy right tail.
- Quantiles: p50 = p75 = 0; p90 = 1.0; p95 = 5.0; p99 = 9.0; p99.9 = 17.0.
- **Zero proportion: 85.49%** (770,073 of 900,797 minutes) — quantifies the
  Phase-1 impression.
- Mean daily volume: **812 L/day**.
- `inst_rate` (not modeled): mean 0.749, max 30.
- *(Interpretation)* The target is a spike train: consumption is rare, short,
  and heavy-tailed when it occurs. Squared-error models will be dominated by
  predicting ~0; zero-inflation must be addressed explicitly (e.g., two-stage
  occurrence/amount models as a later option — *(hypothesis)*, not yet tested).

### Consumption event structure — new measurement

An "event" = maximal run of consecutive minutes with `avg_rate > 0`.

- **48,988 events** over 625.55 days ≈ **78.3 events/day**.
- Duration (minutes): median 2, mean 2.67, p90 5, p99 16, max 94.
- Volume per event (liters): median 3.5, mean 10.4, p90 21, p99 112,
  max 1,165.5.
- Active minutes: 14.51% (complement of the zero proportion).
- *(Interpretation)* Median event volume (~3.5 L) is consistent with single
  fixture uses (toilet flush, short tap); the heavy tail (p99 = 112 L) with
  showers/baths/laundry; the maximum event (1,165 L over ≤94 min) with
  prolonged use such as irrigation. Plausible labels, not verified ground
  truth.

### Temporal patterns (sharpens the Phase-1 profiles)

Hour of day (hours are UTC; local ≈ UTC−7 in PDT, UTC−8 in PST; the profile
mixes both, blurring by ±1 h):

- Deep minimum at 09–10 UTC (≈ 01:00–03:00 local): 0.009–0.014 L/min.
- Strong evening peak at 02–04 UTC (≈ 18:00–21:00 local): up to 1.77 L/min.
- Secondary morning shoulder at 13–16 UTC (≈ 05:00–09:00 local): ~0.23–0.47.
- Non-zero fraction follows the same shape (0.3% at the minimum vs 34% at the
  peak).

Day of week / weekend:

- Day-of-week means are nearly flat (0.530–0.602 L/min; Tuesday lowest, Sunday
  highest); non-zero fractions span only 14.0%–15.0%.
- Weekend vs weekday: mean 0.591 vs 0.553 L/min; non-zero 14.95% vs 14.34%.
- *(Interpretation)* Day-of-week carries very little signal compared to
  hour-of-day; calendar features should derive most of their value from the
  diurnal cycle. To be confirmed by the feature ablation.

### Autocorrelation, first pass — new measurement

ACF of `avg_rate` on the V100 series (FFT-based, biased estimator), and of the
binary indicator `avg_rate > 0`:

| Lag (min) | Meaning | ACF avg_rate | ACF indicator |
|----------:|:--------|-------------:|--------------:|
| 1 | 1 minute | 0.684 | 0.562 |
| 5 | 5 minutes | 0.350 | 0.298 |
| 15 | 15 minutes | 0.139 | 0.186 |
| 30 | 30 minutes | 0.101 | 0.140 |
| 60 | 1 hour | 0.055 | 0.088 |
| 1440 | 1 day | 0.077 | 0.107 |
| 10080 | 1 week | 0.087 | 0.115 |

- Strong short-range dependence decaying within ~1 hour, plus bumps at the
  daily and weekly lags above the surrounding decay.
- The indicator ACF shows the same structure with a slower mid-range decay —
  consistent with events lasting several minutes.

### What Phase 2 changed relative to Phase 1

- Confirmed exactly: no missing values, 1-minute interval, row counts, V100
  cutoff, large zero share, flat day-of-week, visible diurnal structure.
- Newly established: gap-free grid guarantee, `counter` ↔ `avg_rate` identity,
  pulse-size verification, the single 0.053 L artifact, event statistics,
  quantiles, mean daily volume, the full ACF picture.
- Corrected: the old-vs-V100 mean comparison (seasonality confound) and the
  naive reading of timestamps (timezone — resolved in Phase 3).

## Phase 3 — Weather detour and the timezone finding

Goal at the time: assess `Climate_HourlyWeather.csv` as a candidate feature
source. A weather EDA script was built, then **weather was dropped from scope**
(kept the project focused on temporal autocorrelation; the scripts and the
AGENTS.md entries were removed accordingly).

- **Lasting finding from this phase:** `unix_ts` is true Unix/UTC time; the
  derived datetimes are naive UTC. Evidence collected then:
  - the water record starts exactly 7 h after the weather labels start
    (2012-04-01 07:00 UTC vs 2012-04-01 00:00 local; April is PDT = UTC−7), so
    both series begin at the same physical instant;
  - the diurnal minimum of consumption falls at 09–10 in the UTC frame ≈
    01:00–03:00 local — a plausible nighttime minimum; a local-time
    interpretation would put it mid-morning, which is implausible.
- Consequence (kept even though weather is out): hour-of-day features and
  profiles must be converted to local time (America/Vancouver), timezone-aware,
  because the V100 window contains four DST transitions (2012-11-04,
  2013-03-10, 2013-11-03, 2014-03-09).
- Caveat: the V100 window does not start at midnight (17:43 UTC), so the first
  and last days are partial — relevant only for daily aggregations, not for
  minute-level modeling.

## Phase 4 — Autocorrelation deep-dive and feature derivation

Goal: understand the ACF interactively and derive a justified feature set.
Interactive walkthrough in `notebooks/autocorrelation_and_features.py`.

### Analysis steps and what they showed

- ACF computed **directly from the definition** at the candidate lags agrees
  with the FFT-based values of Phase 2 to machine precision (cross-check of
  both implementations).
- **Lag scatter plots** (`y_t` vs `y_{t-k}`, k ∈ {1, 5, 15, 60, 1440}): the
  lag-1 cloud hugs the diagonal (ongoing events persist); larger lags collapse
  toward the origin with structure in the "arms"; the (0,0) mass makes the
  zero-inflation visually explicit.
- **Occurrence process** (indicator `avg_rate > 0`): its ACF decays more slowly
  in the 15–60 min range than the raw ACF — the event-duration effect. Recent
  occurrence is predictable → motivates an `occ_lag_1` feature now and a
  possible two-stage model later (two-stage remains a *(hypothesis)*).
- **Daily/weekly bumps re-checked on hourly means** (less zero-inflation
  distortion): ACF(24 h) = 0.348, ACF(48 h) = 0.340, ACF(72 h) = 0.336,
  ACF(96 h) = 0.328, ACF(168 h) = 0.383, ACF(336 h) = 0.353. Daily harmonics
  and the weekly peak are clearly present on the aggregated series; seasonal
  contamination is reduced but not eliminated.

### Timezone verified on the data

- Localizing the index as UTC and converting to America/Vancouver yields
  UTC↔local offsets of exactly **7 (PDT) and 8 (PST) hours only** — consistent
  with the Phase-3 finding.

### Derived candidate feature set (leakage-safe construction)

- Every feature uses only information from `t-1` or earlier (shift semantics;
  asserted against the raw series in the notebook).
- Set, in priority order, each justified by a measurement above:
  1. Recent consumption lags: `lag_1, lag_5, lag_15, lag_30, lag_60`
     (ACF decay region + event durations)
  2. Daily/weekly lags: `lag_1440, lag_10080` (24 h/168 h bumps)
  3. Occurrence lag: `occ_lag_1` (indicator ACF)
  4. Calendar: `hour_sin, hour_cos` (diurnal profile), `dow_local, weekend`
     (cheap; expected weak)
- **Deliberately excluded**: rolling statistics (partially redundant with the
  lags they average over; revisit only with methodological justification) and
  weather (Phase 3 scope decision).
- **Known issue for the modeling notebook**: `lag_10080` produces a 7-day
  warm-up NaN tail (10,080 rows out of 900,797). The chronological split must
  place that tail inside the training region, never at a split boundary.

## Decisions log (chronological)

1. Analysis restricted to the V100 meter period — *Phase 1, verified Phase 2*.
2. Target = `avg_rate` (interval consumption); `counter` only for validation —
   *Phase 1/2*.
3. Weather excluded from scope (along with its EDA scripts) — *Phase 3*.
4. Rolling statistics excluded from the feature set — *Phase 4*.
5. Forecasting horizon: still open (working assumption: one-step-ahead, h=1).

## Implications for modeling (hypotheses, to be tested)

- The persistence baseline (`y_hat_t = y_{t-1}`) will be a strong opponent
  given ACF(1) = 0.68 and the event structure.
- Zero-inflation favors metrics/analyses that treat occurrence separately from
  amount, or at minimum reporting regression metrics alongside
  occurrence-related behavior.
- Expected feature-importance ordering *(hypothesis)*: recent lags (1–60)
  first, hour-of-day next, daily/weekly lags next, day-of-week/weekend last.
- Exploiting autocorrelation = lag features; mitigating leakage = chronological
  splits and features computed only from `t-1` or earlier. The ablation across
  feature groups (calendar → +recent lags → +daily/weekly lags) directly
  measures how much performance comes from temporal dependence.

## Open items / next steps

1. Finalize the forecasting horizon (working assumption: h = 1, one-step-ahead).
2. Fix chronological train/validation/test split sizes.
3. Turn the Phase-4 feature frame into the leakage-safe modeling pipeline
   (calendar in local time with cyclic encodings; lags
   1, 5, 15, 30, 60, 1440, 10080; `occ_lag_1`).
4. Implement baselines: persistence, previous-day, plus mean/zero references.
5. First ML model (LightGBM) and the feature-group ablation.
6. Diagnostics: feature importance, residual ACF, small-scale hyperparameter
   tuning.

---

## Appendix: Reproducibility

- Phase 1 — personal first inspection: `notebooks/explore_whw.py`
  (exploratory, not part of the pipeline).
- Phase 2 — systematic EDA: `scripts/eda_whw.py`, rerun with
  `uv run python scripts/eda_whw.py`. Regenerates:
  - `data/processed/whw_v100.parquet` (cleaned V100 period)
  - `results/eda_whw/summary.json` (all numbers cited above)
  - `results/eda_whw/figs/` (distribution, time series, transition zoom,
    diurnal/weekly profiles, ACF)
- Phase 4 — autocorrelation/feature walkthrough:
  `notebooks/autocorrelation_and_features.py` (interactive; verifies the ACF,
  examines the occurrence process, and derives the candidate feature set with
  leakage-safe construction).
- Environment: Python 3.12, dependencies pinned via `uv` (see `pyproject.toml`;
  ACF computed with numpy FFT, no statsmodels needed; `tzdata` available for
  local-time conversion).
