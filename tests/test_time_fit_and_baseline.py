import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from calibration import CalibrationResult
from fitting import FitResult, FitParams


def _write_cfg(tmp_path, baseline_range):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": baseline_range, "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"expected_peaks": {"Po210": 0}},
        "analysis": {"analysis_start_time": 0, "analysis_end_time": 5},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0, 20],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


def _write_data(tmp_path):
    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3],
            "fBits": [0, 0, 0],
            "timestamp": [
                pd.Timestamp(0.5, unit="s", tz="UTC"),
                pd.Timestamp(1.5, unit="s", tz="UTC"),
                pd.Timestamp(2.5, unit="s", tz="UTC"),
            ],
            "adc": [8.0, 8.0, 8.0],
            "fchannel": [1, 1, 1],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _patch_common(monkeypatch):
    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={"Po210": {"centroid_adc": 10}},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())


def test_two_pass_time_fit_used(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path, [0, 5])
    data_path = _write_data(tmp_path)

    _patch_common(monkeypatch)

    called = {}

    def fake_two_pass(*args, **kwargs):
        called["ok"] = True
        return FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "two_pass_time_fit", fake_two_pass)

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
    analyze.main()

    assert called.get("ok")


def test_baseline_validation_fails_fast(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path, [10, 20])
    data_path = _write_data(tmp_path)

    _patch_common(monkeypatch)

    def fake_two_pass(*a, **k):  # should not be called
        raise AssertionError("two_pass_time_fit should not run")

    monkeypatch.setattr(analyze, "two_pass_time_fit", fake_two_pass)

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
