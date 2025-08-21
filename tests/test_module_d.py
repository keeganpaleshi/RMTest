import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze
import baseline_noise
from calibration import CalibrationResult
from fitting import FitResult, FitParams


def _prepare_basic_cfg(tmp_path, baseline_range=None, analysis_window=None):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {
            "range": baseline_range,
            "monitor_volume_l": 605.0,
            "sample_volume_l": 0.0,
        },
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": False,
            "expected_peaks": {"Po210": 0},
        },
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
    if analysis_window:
        cfg["analysis"] = {
            "analysis_start_time": analysis_window[0],
            "analysis_end_time": analysis_window[1],
        }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    return cfg_path


def _prepare_basic_data(tmp_path):
    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3],
            "fBits": [0, 0, 0],
            "timestamp": [
                pd.Timestamp(0.5, unit="s", tz="UTC"),
                pd.Timestamp(2.5, unit="s", tz="UTC"),
                pd.Timestamp(3.5, unit="s", tz="UTC"),
            ],
            "adc": [8.0, 8.0, 8.0],
            "fchannel": [1, 1, 1],
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    return data_path


def _patch_common(monkeypatch, tmp_path):
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
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "write_summary", lambda out_dir, summary, timestamp=None: str(Path(out_dir) / "x"))


def test_two_pass_time_fit_used(tmp_path, monkeypatch):
    cfg_path = _prepare_basic_cfg(tmp_path, baseline_range=[0, 5])
    data_path = _prepare_basic_data(tmp_path)
    _patch_common(monkeypatch, tmp_path)

    called = {}

    def fake_two_pass(ts_dict, t_start, t_end, cfg, **kwargs):
        called["yes"] = True
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    def fail_fit(*a, **k):
        raise AssertionError("fit_time_series called directly")

    monkeypatch.setattr(analyze, "two_pass_time_fit", fake_two_pass)
    monkeypatch.setattr(analyze, "fit_time_series", fail_fit)

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
    assert called.get("yes")


def test_baseline_validation_fails_fast(tmp_path, monkeypatch):
    cfg_path = _prepare_basic_cfg(
        tmp_path,
        baseline_range=[20, 30],
        analysis_window=(0, 10),
    )
    data_path = _prepare_basic_data(tmp_path)
    _patch_common(monkeypatch, tmp_path)

    called = {}

    def fake_two_pass(*a, **k):
        called["called"] = True
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

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
    assert "called" not in called
