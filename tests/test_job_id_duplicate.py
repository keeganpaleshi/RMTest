import json
import sys
from pathlib import Path

import pytest

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
from fitting import FitResult


def test_duplicate_job_id_raises(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }

    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [0],
        "adc": [1000],
        "fchannel": [1],
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0, 0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    def fake_copy(path, cfg):
        Path(path).mkdir(parents=True, exist_ok=False)

    monkeypatch.setattr(analyze, "copy_config", fake_copy)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
        "--job-id",
        "DUPLICATE",
    ]

    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    # Second run with the same job-id should raise FileExistsError
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(FileExistsError):
        analyze.main()

