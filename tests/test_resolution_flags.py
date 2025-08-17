import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fitting import fit_spectrum
import analyze
from calibration import CalibrationResult


def test_fit_spectrum_conflicting_resolution_flags():
    energies = np.array([1.0, 2.0])
    priors = {"sigma0": (0.05, 0.01), "F": (0.0, 0.01)}
    with pytest.raises(ValueError):
        fit_spectrum(energies, priors, flags={"fix_sigma0": True, "fix_F": False})


def test_analyze_config_resolution_conflict(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": True,
            "float_sigma_E": True,
            "flags": {"fix_sigma0": True},
            "expected_peaks": {"Po210": 0, "Po218": 0, "Po214": 0},
            "amp_prior_scale": 1.0,
        },
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")],
            "adc": [1],
            "fchannel": [1],
        }
    )
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
    monkeypatch.setattr(analyze, "find_adc_bin_peaks", lambda *a, **k: cfg["spectral_fit"]["expected_peaks"])
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(ValueError):
        analyze.main()
