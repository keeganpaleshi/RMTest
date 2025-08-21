import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure the repository root is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
from calibration import CalibrationResult
from fitting import FitResult, FitParams


def _setup_common(monkeypatch):
    """Patch heavy functions with lightweight stand-ins."""
    cal = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal)
    monkeypatch.setattr(
        analyze,
        "fit_spectrum",
        lambda *a, **k: FitResult(FitParams({}), np.zeros((0, 0)), 0),
    )
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(
        analyze,
        "plot_time_series",
        lambda *a, **k: Path(k["out_png"]).touch(),
    )
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "plot_equivalent_air", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_radon_activity_full", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_radon_trend_full", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "write_summary", lambda *a, **k: str(Path(a[0])))


def _write_input(tmp_path, cfg, df):
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    return cfg_path, data_path


def test_two_pass_time_fit_used(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0, 2000],
            "flags": {},
        },
        "analysis_isotope": "po214",
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"], "plot_time_binning_mode": "fixed"},
    }
    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")],
            "adc": [1000],
            "fchannel": [1],
        }
    )
    cfg_path, data_path = _write_input(tmp_path, cfg, df)

    _setup_common(monkeypatch)

    calls = {"tp": 0}

    def fake_two_pass(*a, **k):
        calls["tp"] += 1
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "two_pass_time_fit", fake_two_pass)

    def fail_direct(*a, **k):
        raise AssertionError("fit_time_series should not be called directly")

    monkeypatch.setattr(analyze, "fit_time_series", fail_direct)

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
    assert calls["tp"] > 0


def test_baseline_validation_fails_fast(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {
            "analysis_start_time": "1970-01-01T00:00:00Z",
            "analysis_end_time": "1970-01-01T01:00:00Z",
        },
        "calibration": {},
        "spectral_fit": {"expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "baseline": {
            "range": [
                "1970-01-02T00:00:00Z",
                "1970-01-02T01:00:00Z",
            ]
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"], "plot_time_binning_mode": "fixed"},
    }
    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")],
            "adc": [1000],
            "fchannel": [1],
        }
    )
    cfg_path, data_path = _write_input(tmp_path, cfg, df)

    _setup_common(monkeypatch)

    called = {"v": False}

    def fake_validate(cfg):
        called["v"] = True
        raise ValueError("bad baseline")

    monkeypatch.setattr(analyze, "validate_baseline_window", fake_validate)

    def forbid(*a, **k):  # pragma: no cover - should not be called
        raise AssertionError("baseline subtraction should not run")

    monkeypatch.setattr(analyze, "subtract_baseline_counts", forbid)
    monkeypatch.setattr(analyze, "subtract_baseline_rate", forbid)

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
    assert called["v"]
