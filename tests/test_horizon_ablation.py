import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from horizon_ablation import add_features, aggregate_target


def test_aggregate_target_excludes_incomplete_edge_blocks():
    timestamps = np.arange(120, 32 * 60, 60, dtype=np.int64)
    source = pd.DataFrame(
        {"avg_rate": np.arange(1, len(timestamps) + 1, dtype=np.float64)},
        index=pd.Index(timestamps, name="unix_ts"),
    )

    blocks, metadata = aggregate_target(source, 15)

    assert blocks.index.tolist() == [900]
    assert blocks["target_liters"].tolist() == [330.0]
    assert metadata["excluded_incomplete_edge_rows"] == 15


def test_block_lags_use_only_completed_targets():
    horizon = 15
    start = 1_700_000_000 // (horizon * 60) * (horizon * 60)
    timestamps = start + np.arange(700, dtype=np.int64) * horizon * 60
    blocks = pd.DataFrame(
        {"target_liters": np.arange(700, dtype=np.float64)},
        index=pd.Index(timestamps, name="unix_ts"),
    )

    features, lag_blocks = add_features(blocks, horizon)

    assert lag_blocks == [1, 2, 4, 96, 672]
    assert pd.isna(features.loc[timestamps[0], "lag_1_block"])
    assert features.loc[timestamps[1], "lag_1_block"] == 0.0
    assert features.loc[timestamps[2], "lag_2_block"] == 0.0
    assert features.loc[timestamps[672], "lag_672_block"] == 0.0
