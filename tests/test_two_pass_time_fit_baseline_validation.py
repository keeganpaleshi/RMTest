import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from calibration import CalibrationResult
from fitting import FitResult, FitParams


def _write_cfg(tmp_path, cfg):
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    return cfg_path


def _write_data(tmp_path, df):
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    return data_path


def _patch_common(monkeypatch):
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
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)


def test_analyze_uses_two_pass_time_fit(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7, 9],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(1.0, unit="s", tz="UTC")],
            "adc": [8],
            "fchannel": [1],
        }
    )
    data_path = _write_data(tmp_path, df)

    _patch_common(monkeypatch)

    called = {}

    def fake_two_pass(*args, **kwargs):
        called["ok"] = True
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    def fail_fit(*args, **kwargs):
        raise AssertionError("fit_time_series should not be called directly")

    monkeypatch.setattr(analyze, "two_pass_time_fit", fake_two_pass)
    monkeypatch.setattr(analyze, "fit_time_series", fail_fit)

    def fake_write(out_dir, summary, timestamp=None):
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)

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
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"analysis_start_time": 0, "analysis_end_time": 10},
        "baseline": {"range": [20, 30], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7, 9],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(1.0, unit="s", tz="UTC")],
            "adc": [8],
            "fchannel": [1],
        }
    )
    data_path = _write_data(tmp_path, df)

    _patch_common(monkeypatch)

    def fail_two_pass(*args, **kwargs):
        raise AssertionError("two_pass_time_fit should not be called")

    monkeypatch.setattr(analyze, "two_pass_time_fit", fail_two_pass)
    monkeypatch.setattr(
        analyze.baseline,
        "subtract",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("baseline subtraction should not run")),
    )

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
