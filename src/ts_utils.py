"""
Small, reusable time-series utilities shared across scripts in this project.

Kept separate from any single analysis script (e.g. eda_whw.py) so that
functions used in more than one place -- the FFT-based ACF, and the boolean
run-length segmentation used for event/gap detection -- have a single
implementation, instead of being copy-pasted between scripts and risking
drift. In particular, these are expected to be reused once the project
moves to the 15-minute aggregated resolution (see AGENTS.md, "Forecasting
Horizon Decision"), where the ACF needs to be recomputed on the resampled
series using the same method as the original 1-minute EDA.
"""

import numpy as np
import pandas as pd


def to_serializable(obj):
    """
    Recursively convert numpy/pandas scalar and container types into plain
    Python types that json.dump can handle natively.

    Needed because dicts built from pandas/numpy operations are full of
    np.int64, np.float64, np.bool_, and pd.Timestamp values, none of which
    the standard library json module knows how to serialize on its own.
    """
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def acf_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute the sample autocorrelation function of x for lags 0..max_lag,
    using an FFT-based autocovariance instead of a direct/naive lag loop.

    Why FFT: computing the autocovariance directly at every lag up to
    max_lag is O(n * max_lag); for this project n ~ 1e6 rows and
    max_lag = 10080 (one week of 1-minute data), which would be far too
    slow. The FFT approach computes the full autocovariance in O(n log n)
    by zero-padding to at least 2n - 1 samples (avoiding circular-
    correlation wraparound) and using the convolution theorem: the
    autocovariance is the convolution of the mean-centered series with its
    own time-reversal, which the FFT computes efficiently.

    Returns the *normalized* ACF (acf[0] == 1.0 whenever the series has
    nonzero variance; an all-constant series returns all zeros rather than
    dividing by zero).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()  # next power of two >= 2n - 1
    f = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(f * f.conj(), n=nfft)[: max_lag + 1]
    denom = float((x * x).sum())
    return acov / denom if denom > 0 else np.zeros(max_lag + 1)


def segment_runs(mask: np.ndarray):
    """
    Find contiguous runs of True values in a 1D boolean array.

    Returns (starts, ends): integer index arrays such that, for every run
    i, mask[starts[i]:ends[i]] is entirely True (half-open interval, so
    ends[i] - starts[i] is the run's length in samples).

    Method: take the first difference of the boolean array (as int8). A +1
    marks a False->True transition (a run start); a -1 marks a True->False
    transition (a run end). np.diff can't see past the array's own edges,
    so the two boundary cases -- the array already being "inside" a run at
    index 0, or still inside one at the last index -- are patched in
    explicitly.

    Generic over what "True" means: used both for water-use *events*
    (True = avg_rate > 0) and, symmetrically, for zero-flow *gaps* between
    events (True = avg_rate == 0), simply by passing a different mask.
    """
    mask = np.asarray(mask)
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    return starts, ends


def distribution_summary(x) -> dict:
    """
    Fixed-shape summary (count, mean, median, p90, p99, max) for a 1D array
    of values. Used for any per-run quantity -- event durations, event
    volumes, gap lengths, etc. -- so different quantities are reported with
    the same set of statistics and are easy to compare side by side.
    """
    x = np.asarray(x, dtype=np.float64)
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }