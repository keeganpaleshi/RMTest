import sys
from pathlib import Path
import json
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import numpy as np
from calibration import CalibrationResult


def test_time_bin_mode_conflict(tmp_path, monkeypatch):
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

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data_path),
        "--output_dir", str(tmp_path),
        "--plot-time-binning-mode", "auto",
        "--time-bin-mode", "fd",
    ]
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(SystemExit):
        analyze.main()

