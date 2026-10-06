"""Run the full pipeline in order; prediction plots are made separately with src/plot_predictions.py."""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
SCRIPTS = [
    "eda.py",
    "build_features.py",
    "baselines.py",
    "compare_models.py",
    "feature_ablation.py",
    "feature_importance.py",
    "residual_acf.py",
    "test_evaluation.py",
]

for script in SCRIPTS:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / script)], check=True)
