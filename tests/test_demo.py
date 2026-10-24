import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_installed_example_runs_from_another_directory(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-m", "hmm_ids.demo", "--output-dir", "demo", "--seed", "42"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "demo/summary.json").read_text())
    assert summary["data_kind"] == "synthetic_categorical_hmm"
    assert summary["observations"] == 400
    assert np.isfinite(summary["known_model_log_likelihood"])
    assert np.min(np.diff(summary["em_history"])) >= -1e-9
    assert summary["filtered_row_sum_max_error"] < 1e-12
    assert (tmp_path / "demo/inference-demo.png").stat().st_size > 1000
