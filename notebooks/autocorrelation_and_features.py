# %% [markdown]
# # Autocorrelation and Feature Selection — Walkthrough
#
# This notebook walks step by step through the autocorrelation analysis of the
# whole-house water series (V100 meter period) and derives a candidate feature
# set from what we actually measured. It complements `scripts/eda_whw.py`
# (systematic EDA) and `REPORT.md` (accumulated findings).
#
# Everything below is computed live from `data/processed/whw_v100.parquet`,
# so you can verify every number yourself.

# %% Imports and configuration
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path("../data/processed/whw_v100.parquet")

# Candidate lag set from AGENTS.md (in minutes).
CANDIDATE_LAGS = [1, 5, 15, 30, 60, 1440, 10080] # 60=1h, 1440=1d, 10080=1w

# The house is in Burnaby, BC, Canada. Our EDA established that `unix_ts`
# is true UTC time, so local-time features need an explicit conversion.
LOCAL_TZ = "America/Vancouver"

# %% Load the cleaned V100 series
df = pd.read_parquet(DATA_PATH)

# The modeling target: interval consumption in L/min at minute t.
y = df["avg_rate"]

print(f"Rows:  {len(df):,}")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")
print(f"Zero proportion: {(y == 0).mean():.2%}")
display(df.head())

# %% [markdown]
# ## 1. What is autocorrelation?
#
# The autocorrelation function (ACF) at lag $k$ measures how strongly the
# series at time $t$ is linearly related to the series $k$ steps earlier:
#
# $$\rho(k) = \frac{\sum_{t} (x_t - \bar{x})(x_{t+k} - \bar{x})}
#                     {\sum_{t} (x_t - \bar{x})^2}$$
#
# Why we care (research question 3):
# - **Exploiting autocorrelation**: if $\rho(k)$ is large, then $x_{t-k}$
#   (a *lag feature*) carries predictive information about $x_t$.
# - **Leakage safety**: a feature may only use information from $t-1$ or
#   earlier (working assumption: one-step-ahead, horizon h=1).
#
# First, we compute the ACF at the candidate lags *directly from the
# definition*, so the estimator is not a black box.

# %% ACF at candidate lags, straight from the definition
def acf_direct(x: pd.Series, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x0 = x - x.mean()
    numerator = np.dot(x0[:-lag], x0[lag:])
    denominator = np.dot(x0, x0)
    return numerator / denominator


manual_acf = {lag: acf_direct(y, lag) for lag in CANDIDATE_LAGS}
display(pd.Series(manual_acf, name="ACF (direct)").round(3))

# %% [markdown]
# Observations:
#
# - `rho(1) = 0.68`: the single strongest linear relationship. This is why the
#   **persistence baseline** (`y_hat_t = y_{t-1}`) will be a serious opponent.
# - The decay is fast: by 60 minutes the correlation has dropped to ~0.055.
# - But at 1440 (one day) and 10080 (one week) the correlation *rises again*
#   above the surrounding decay — that is the signature of daily and weekly
#   seasonality.
#
# Computing this one lag at a time is fine for 7 lags, but for a full curve we
# use the FFT — mathematically the same estimator, just computed for all lags
# at once. Let's verify both implementations agree.

# %% FFT-based ACF, verified against the direct computation
def acf_fft(x: pd.Series, max_lag: int) -> np.ndarray:
    """ACF for all lags 0..max_lag via FFT (same estimator as eda_whw.py)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(f * f.conj(), n=nfft)[: max_lag + 1]
    return acov / float((x * x).sum())


# Computed out to the weekly lag (10080) so every candidate lag is covered;
# plots below show the first 2 days only.
acf = acf_fft(y, max_lag=10080)

fft_acf = {lag: float(acf[lag]) for lag in CANDIDATE_LAGS}
comparison = pd.DataFrame({"direct": pd.Series(manual_acf), "fft": pd.Series(fft_acf)})
comparison["abs_diff"] = (comparison["direct"] - comparison["fft"]).abs()
display(comparison.round(6))

# %%
acf_display_len = 2880
lags = np.arange(acf_display_len + 1)
acf_to_plot = acf[: acf_display_len + 1]

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
axes[0].plot(lags, acf_to_plot, linewidth=0.8)
axes[0].set_title("ACF of avg_rate — first 2 days")
axes[0].set_ylabel("ACF")

zoom = 120
axes[1].plot(lags[: zoom + 1], acf_to_plot[: zoom + 1], marker="o", markersize=3)
axes[1].set_title("Zoom: first 2 hours (the decay region)")
axes[1].set_xlabel("Lag (minutes)")
axes[1].set_ylabel("ACF")

for ax in axes:
    ax.axhline(0, color="black", linewidth=0.5)
    for lag in CANDIDATE_LAGS:
        if lag <= len(acf) - 1:
            ax.axvline(lag, color="red", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# How to read this:
#
# - Left region (zoom): a steep decay over the first hour. Consecutive minutes
#   belong to the same *consumption event* (median event duration is ~2 min),
#   so nearby values are strongly dependent.
# - Big picture: at lag 1440 there is a clear bump — yesterday's pattern
#   predicts today's better than the decayed short-term memory would suggest.
#   This is the daily routine of the household.
# - Lag 10080 (one week) is beyond this plot; we verify it more carefully
#   below via hourly aggregation.
#
# One honest caveat before moving on: with 85.5% zeros, the raw-value ACF is
# largely driven by the *occurrence* process (is water flowing at all?). We
# examine that process explicitly in section 3.

# %% [markdown]
# ## 2. Lag scatter plots — what a lag feature literally "sees"
#
# A lag feature is just the series shifted. These scatter plots show the pairs
# $(x_{t-k}, x_t)$ that the model will be fed. Note how the ACF value is
# exactly the Pearson correlation of these two vectors — one number summarizing
# a whole cloud of points.

# %%
fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)

for ax, lag in zip(axes, [1, 5, 15, 60, 1440]):
    x_prev = y[:-lag]
    x_now = y[lag:]

    # Subsample for plotting; the correlation is computed on the full vectors.
    ax.scatter(x_prev[::10], x_now[::10], s=1, alpha=0.05, color="steelblue")
    r = np.corrcoef(x_prev, x_now)[0, 1]
    ax.set_title(f"lag = {lag} min\nPearson r = {r:.3f}")
    ax.set_xlabel(f"y(t - {lag})")
    ax.set_xlim(0, 20)
    ax.set_aspect("equal")

axes[0].set_ylabel("y(t)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Reading the clouds:
#
# - At lag 1 the mass hugs the diagonal: if the meter registered flow a minute
#   ago, it very likely still does. That diagonal is what persistence exploits.
# - As the lag grows the cloud collapses toward the origin — most minutes are
#   zero at both ends. The visible "arms" are ongoing events.
# - At lag 1440 the cloud is sparse but not structureless: active minutes tend
#   to pair with active minutes (routine).
#
# The zero-inflation is visible everywhere: the vast majority of points sit at
# (0, 0). Keep this in mind — a model that predicts ~0 everywhere already gets
# most minutes right by accident, which is why we will need better evaluation
# views than RMSE alone.

# %% [markdown]
# ## 3. The occurrence process — predictability of "is water flowing?"
#
# Because 85.5% of minutes are zero, we split the problem mentally into:
#
# 1. **Occurrence**: is $y_t > 0$?
# 2. **Amount**: given flow, how much?
#
# If the occurrence process is itself autocorrelated, occurrence is predictable
# from recent history — the basis for a possible two-stage model later
# (hypothesis for now, not yet built).

# %%
occ = (y > 0).astype(float)
acf_occ = acf_fft(occ, max_lag=10080)

plot_len = 2880
lags_axis = np.arange(plot_len + 1)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(lags_axis, acf_to_plot, linewidth=0.8, label="ACF of avg_rate")
ax.plot(lags_axis, acf_occ[: plot_len + 1], linewidth=0.8, color="orange", label="ACF of (avg_rate > 0)")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Lag (minutes)")
ax.set_ylabel("ACF")
ax.set_title("Raw-value ACF vs occurrence (indicator) ACF")
ax.legend()

for lag in CANDIDATE_LAGS:
    if lag <= len(acf) - 1:
        ax.axvline(lag, color="red", linestyle=":", alpha=0.4)

plt.tight_layout()
plt.show()

comparison_occ = pd.DataFrame(
    {
        "avg_rate": [acf[l] for l in CANDIDATE_LAGS],
        "indicator": [acf_occ[l] for l in CANDIDATE_LAGS],
    },
    index=CANDIDATE_LAGS,
)
display(comparison_occ.round(3))

# %% [markdown]
# Both curves show the same structure, but the indicator decays more slowly in
# the mid range (15–60 min). That is the event-duration effect: once flow has
# started it tends to persist for several minutes (median 2, p99 = 16).
#
# Practical consequence: an `occ_lag_1` feature (was there flow one minute
# ago?) is nearly free to compute and directly targets the dominant signal.
# The full two-stage model remains a *hypothesis* for later; the feature set
# below only borrows the cheap part.

# %% [markdown]
# ## 4. Daily and weekly structure, checked more cleanly
#
# The raw-minute ACF mixes the occurrence and amount processes. To examine the
# daily/weekly bumps with less zero-inflation distortion, we aggregate to
# **hourly means** first and compute the ACF on that series:
#
# - 24 lags of the hourly series = 1 day
# - 168 lags = 1 week
#
# Caveat (same as REPORT.md): long-lag ACF on a seasonal series is partly
# contaminated by seasonality itself — the hourly curve supports the choice of
# daily/weekly lags, but the final arbiter is the feature ablation.

# %%
y_hourly = y.resample("1h").mean()
acf_hourly = acf_fft(y_hourly, max_lag=336)

hours_axis = np.arange(len(acf_hourly))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(hours_axis, acf_hourly, linewidth=0.8)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Lag (hours)")
ax.set_ylabel("ACF of hourly-mean consumption")
ax.set_title("Hourly ACF — daily (24 h) and weekly (168 h) bumps")

for lag in [24, 48, 72, 96, 168, 336]:
    ax.axvline(lag, color="red", linestyle=":", alpha=0.4)

plt.tight_layout()
plt.show()

display(
    pd.Series(
        {f"{lag} h ({lag // 24:.0f} d)": acf_hourly[lag] for lag in [24, 48, 72, 96, 168, 336]},
        name="ACF of hourly means",
    ).round(3)
)

# %% [markdown]
# The bumps at 24 h (and harmonics 48/72/96 h) are unmistakable, and there is a
# smaller but real peak at 168 h — weekly structure exists.
#
# Measured motivation summary so far:
#
# | EDA measurement | Feature it motivates |
# |---|---|
# | ACF(1)=0.68 decaying through lag 60 | recent lags: 1, 5, 15, 30, 60 |
# | indicator ACF mid-range persistence | occurrence lag 1 (cheap two-stage signal) |
# | 24 h bump (raw and hourly ACF) | lag 1440 |
# | 168 h bump (hourly ACF) | lag 10080 |
# | diurnal profile: 0.3% vs 34% non-zero | hour-of-day (cyclic sin/cos) |
# | day-of-week nearly flat (~5% diff) | dow/weekend — cheap, low expectation |
# | *(hypothesis)* | rolling statistics — **excluded** for now per scope decision |

# %% [markdown]
# ## 5. Building the candidate feature frame — leakage-safe
#
# Rules encoded below (these implement the no-leakage constraints from
# AGENTS.md):
#
# - The target is `y_t` (this minute's consumption).
# - Every feature is built with `shift(k)`, `k >= 1`: the newest information
#   any feature may use is from `t-1`. Nothing about minute t may leak in.
# - Calendar features are computed in **local time** (`America/Vancouver`),
#   because the household's behavior follows local clocks, not UTC.
#   Our EDA established the naive index is true UTC, so we first *localize*
#   (declare UTC) and then *convert*; pandas handles the DST transitions.
# - Hour-of-day is encoded cyclically (sin/cos) so that 23:00 and 00:00 are
#   neighbors, as they are in reality.

# %%
feats = pd.DataFrame(index=df.index)
feats["target"] = y

for lag in CANDIDATE_LAGS:
    feats[f"lag_{lag}"] = y.shift(lag)

feats["occ_lag_1"] = (y > 0).astype(int).shift(1)

local_index = df.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
feats["hour_local"] = local_index.hour
feats["dow_local"] = local_index.dayofweek
feats["weekend"] = (feats["dow_local"] >= 5).astype(int)
feats["hour_sin"] = np.sin(2 * np.pi * feats["hour_local"] / 24)
feats["hour_cos"] = np.cos(2 * np.pi * feats["hour_local"] / 24)

print(f"Feature frame: {feats.shape[0]:,} rows x {feats.shape[1]} columns")
feats.head()

# %% Sanity check 1: the shifts really reference t-1 and earlier
i = 500_000
assert feats["lag_1"].iloc[i] == y.iloc[i - 1]
assert feats["lag_5"].iloc[i] == y.iloc[i - 5]
assert feats["lag_1440"].iloc[i] == y.iloc[i - 1440]
assert feats["occ_lag_1"].iloc[i] == int(y.iloc[i - 1] > 0)
print("All shift checks passed: features only see t-1 or earlier.")

# %% Sanity check 2: UTC vs local hours differ by 7 or 8 (DST)
tz_check = pd.DataFrame(
    {
        "utc_hour": df.index.hour,
        "local_hour": local_index.hour,
    },
    index=df.index,
)
tz_check["offset"] = (tz_check["utc_hour"] - tz_check["local_hour"]) % 24
display(tz_check["offset"].value_counts().sort_index())
# Expect only offsets 7 (PDT, summer) and 8 (PST, winter).

# %% Sanity check 3: missing values only from the longest lag
print("NaN counts per column (only the longest lag should have a warm-up tail):")
print(feats.isna().sum().to_string())

# %% Visual sanity check: target vs its two strongest lags
window = feats.iloc[20_000 : 20_000 + 3 * 1440]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(window.index, window["target"], linewidth=0.8, label="target y(t)")
ax.plot(window.index, window["lag_1"], linewidth=0.8, alpha=0.6, label="lag 1 (persistence input)")
ax.plot(window.index, window["lag_1440"], linewidth=0.8, alpha=0.6, label="lag 1440 (previous day)")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("avg_rate (L/min)")
ax.set_title("Target vs lag-1 vs lag-1440 over 3 days")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The lag-1 trace is the target shifted right by one minute (identical event
# shapes); the lag-1440 trace outlines the previous day's routine.
#
# ## 6. Takeaways
#
# **Candidate feature set carried into modeling** (each justified by a
# measurement above, in priority order):
#
# 1. Recent consumption lags: `lag_1, lag_5, lag_15, lag_30, lag_60`
# 2. Daily/weekly lags: `lag_1440, lag_10080`
# 3. Occurrence lag: `occ_lag_1`
# 4. Calendar: `hour_sin, hour_cos` (primary), `dow_local, weekend` (cheap,
#    expected weak)
#
# **Deliberately excluded**: rolling statistics (scope decision — partially
# redundant with the lags they average over; revisit only with a
# methodological justification, per AGENTS.md).
#
# **Known issue for the modeling notebook**: `lag_10080` produces a 7-day
# warm-up NaN tail (10,080 rows out of 900,797). The chronological split must
# place that tail inside the training region, never at a split boundary.
#
# **Still open before any modeling**: fix the forecasting horizon (working
# assumption: h = 1) and the chronological train/validation/test sizes.
