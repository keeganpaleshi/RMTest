import sys
from pathlib import Path
import json
from typing import Any
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import radon_activity
import radon_joint_estimator
import numpy as np
from dataclasses import asdict
from calibration import CalibrationResult
from fitting import FitResult, FitParams


def test_total_radon_uses_sample_volume(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"monitor_volume_l": 10.0, "sample_volume_l": 5.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": True, "window_po214": [7, 9], "hl_po214": [1.0, 0.0], "eff_po214": [1.0, 0.0], "flags": {}},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }

    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")],
        "adc": [8],
        "fchannel": [1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={"Po210": {"centroid_adc": 10}},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    monkeypatch.setattr(radon_activity, "compute_radon_activity", lambda *a, **k: (5.0, 0.5))

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data_path),
        "--output_dir", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    summary = captured.get("summary", {})
    total_entry = summary["radon_results"]["total_radon_in_sample_Bq"]
    assert total_entry["value"] == pytest.approx(5.0)
    assert total_entry["uncertainty"] == pytest.approx(0.5)


def test_total_radon_series_background_run(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"monitor_volume_l": 10.0, "sample_volume_l": 0.0},
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

    cfg_path = tmp_path / "cfg_bg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")],
            "adc": [8],
            "fchannel": [1],
        }
    )
    data_path = tmp_path / "bg.csv"
    df.to_csv(data_path, index=False)

    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={"Po210": {"centroid_adc": 10}},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(
        analyze,
        "fit_time_series",
        lambda *a, **k: FitResult(FitParams({}), np.zeros((0, 0)), 0),
    )
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    monkeypatch.setattr(radon_activity, "compute_radon_activity", lambda *a, **k: (5.0, 0.5))

    radon_estimate = {
        "isotope_mode": "radon",
        "Rn_activity_Bq": 5.0,
        "stat_unc_Bq": 0.5,
        "components": {},
    }
    monkeypatch.setattr(
        radon_joint_estimator,
        "estimate_radon_activity",
        lambda *a, **k: dict(radon_estimate),
    )

    captured: dict[str, Any] = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

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

    summary = captured.get("summary", {})
    radon_series = summary["radon"]["time_series"]
    total_series = summary["radon"]["total_time_series"]

    assert np.allclose(total_series["activity"], radon_series["activity"])
    assert np.allclose(total_series["error"], radon_series["error"])
