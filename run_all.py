"""Run the full pipeline in order (Stages 1-6); prediction plots are made separately with src/plot_predictions.py."""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STAGES = [
    "eda.py",
    "build_features.py",
    "baselines.py",
    "compare_models.py",
    "feature_ablation.py",
    "feature_importance.py",
    "residual_acf.py",
    "test_evaluation.py",
]

for stage in STAGES:
    print(f"\n=== {stage} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / stage)], check=True)
