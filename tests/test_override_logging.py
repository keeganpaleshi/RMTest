import json
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from fitting import FitResult
from calibration import CalibrationResult


def _minimal_patches(monkeypatch):
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0, 0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)


def test_analysis_end_time_override_logs(tmp_path, monkeypatch, caplog):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"analysis_end_time": "1970-01-01T00:00:05Z"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    _minimal_patches(monkeypatch)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
        "--analysis-end-time",
        "1970-01-01T00:00:06Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    with caplog.at_level(logging.INFO):
        analyze.main()

    assert "analysis.analysis_end_time" in caplog.text


def test_analysis_start_time_override_logs(tmp_path, monkeypatch, caplog):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"analysis_start_time": "1970-01-01T00:00:05Z"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    _minimal_patches(monkeypatch)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
        "--analysis-start-time",
        "1970-01-01T00:00:06Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    with caplog.at_level(logging.INFO):
        analyze.main()

    assert "analysis.analysis_start_time" in caplog.text
