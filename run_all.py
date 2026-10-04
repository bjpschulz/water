"""Run the full pipeline in order: EDA -> features -> baselines -> diagnostic models (Stage 3) -> feature ablation (Stage 4)."""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STAGES = [
    "initial_eda.py",
    "build_features.py",
    "baselines.py",
    "diagnostic_models.py",
    "feature_ablation.py",
]

for stage in STAGES:
    print(f"\n=== {stage} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / stage)], check=True)
