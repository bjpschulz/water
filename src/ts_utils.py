""" Small, reusable time-series utilities shared across scripts in this project. """

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