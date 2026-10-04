""" Small, reusable time-series utilities shared across scripts in this project. """

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Lag set (1-minute units) shared by the ACF figure, the feature table and the
# models: dense 1-15 (ACF is highest there), then hour-scale, daily, weekly.
LAGS = [*range(1, 16), 30, 45, 60, 1440, 10080]


def to_serializable(obj):
    """
    Recursively convert numpy/pandas scalar and container types into plain
    Python types that json.dump can handle natively.

    Needed because dicts built from pandas/numpy operations are full of
    np.int64, np.float64, np.bool_, and pd.Timestamp values.
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


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, allow_nan=False)
        f.write("\n")


def segment_runs(mask: np.ndarray):
    """
    Find contiguous runs of True values in a 1D boolean array.

    Returns (starts, ends): integer index arrays such that, for every run
    i, mask[starts[i]:ends[i]] is entirely True (half-open interval, so
    ends[i] - starts[i] is the run's length in samples).
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
    x = np.asarray(x, dtype=np.float64)
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }
