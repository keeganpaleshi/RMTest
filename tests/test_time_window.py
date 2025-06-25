import sys
import json
from pathlib import Path
import pandas as pd
import pytest
import numpy as np
import dataclasses
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting import FitResult, FitParams

import analyze
import baseline_noise
from calibration import CalibrationResult


def test_time_window_filters_events(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": ["1970-01-01T00:00:00Z", "1970-01-01T00:00:05Z"], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
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
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3, 4],
        "fBits": [0, 0, 0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(2.0, unit="s", tz="UTC"), pd.Timestamp(6.0, unit="s", tz="UTC"), pd.Timestamp(9.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0, 8.0, 8.0],
        "fchannel": [1, 1, 1, 1],
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
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (5.0, {}))

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict.get("Po214", []).tolist()
        return FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    def fake_write(out_dir, summary, timestamp=None):
        if dataclasses.is_dataclass(summary):
            summary = dataclasses.asdict(summary)
        captured["summary"] = summary
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
        "--analysis-end-time",
        "1970-01-01T00:00:06Z",
        "--spike-end-time",
        "1970-01-01T00:00:01Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    summary = captured.get("summary", {})
    assert summary["baseline"]["n_events"] == 2
    assert captured.get("times") == [2.0, 6.0]


def test_invalid_baseline_range_raises(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": ["1970-01-01T00:00:05Z", "1970-01-01T00:00:02Z"], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": False,
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC")],
        "adc": [8.0],
        "fchannel": [1],
    })
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
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

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


def test_time_window_filters_events_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": ["1970-01-01T00:00:00Z", "1970-01-01T00:00:05Z"], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "analysis": {"analysis_end_time": "1970-01-01T00:00:06Z", "spike_end_time": "1970-01-01T00:00:01Z"},
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
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3, 4],
        "fBits": [0, 0, 0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(2.0, unit="s", tz="UTC"), pd.Timestamp(6.0, unit="s", tz="UTC"), pd.Timestamp(9.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0, 8.0, 8.0],
        "fchannel": [1, 1, 1, 1],
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
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (5.0, {}))

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict.get("Po214", []).tolist()
        return FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    def fake_write(out_dir, summary, timestamp=None):
        if dataclasses.is_dataclass(summary):
            summary = dataclasses.asdict(summary)
        captured["summary"] = summary
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
    assert summary["baseline"]["n_events"] == 2
    assert captured.get("times") == [2.0, 6.0]


def test_run_period_filters_events(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": ["1970-01-01T00:00:00Z", "1970-01-01T00:00:01Z"], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "analysis": {"run_periods": [["1970-01-01T00:00:01Z", "1970-01-01T00:00:06Z"]]},
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
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3, 4],
        "fBits": [0, 0, 0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(2.0, unit="s", tz="UTC"), pd.Timestamp(5.0, unit="s", tz="UTC"), pd.Timestamp(7.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0, 8.0, 8.0],
        "fchannel": [1, 1, 1, 1],
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
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (5.0, {}))

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict.get("Po214", []).tolist()
        return FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    def fake_write(out_dir, summary, timestamp=None):
        if dataclasses.is_dataclass(summary):
            summary = dataclasses.asdict(summary)
        captured["summary"] = summary
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
    assert summary["baseline"]["n_events"] == 1
    assert captured.get("times") == [2.0, 5.0]


@pytest.mark.parametrize(
    "start,end",
    [
        ("1970-01-01T00:00:00", "1970-01-01T00:00:05"),
        ("1970-01-01T00:00:00Z", "1970-01-01T00:00:05Z"),
    ],
)
def test_baseline_range_iso_strings(tmp_path, monkeypatch, start, end):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [start, end], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "analysis": {"analysis_end_time": "1970-01-01T00:00:06Z", "spike_end_time": "1970-01-01T00:00:01Z"},
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
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3, 4],
        "fBits": [0, 0, 0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(2.0, unit="s", tz="UTC"), pd.Timestamp(6.0, unit="s", tz="UTC"), pd.Timestamp(9.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0, 8.0, 8.0],
        "fchannel": [1, 1, 1, 1],
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
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (5.0, {}))

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict.get("Po214", []).tolist()
        return FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    def fake_write(out_dir, summary, timestamp=None):
        if dataclasses.is_dataclass(summary):
            summary = dataclasses.asdict(summary)
        captured["summary"] = summary
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
    assert summary["baseline"]["n_events"] == 2
    assert captured.get("times") == [2.0, 6.0]



def test_unified_filter_combined_windows(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": ["1970-01-01T00:00:00Z", "1970-01-01T00:00:01Z"], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "analysis": {
            "run_periods": [["1970-01-01T00:00:01Z", "1970-01-01T00:00:04Z"]],
            "spike_periods": [["1970-01-01T00:00:02Z", "1970-01-01T00:00:02.5Z"]],
        },
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0, 10],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3, 4, 5, 6],
        "fBits": [0, 0, 0, 0, 0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(1.0, unit="s", tz="UTC"), pd.Timestamp(2.0, unit="s", tz="UTC"), pd.Timestamp(2.2, unit="s", tz="UTC"), pd.Timestamp(3.0, unit="s", tz="UTC"), pd.Timestamp(5.0, unit="s", tz="UTC")],
        "adc": [8.0]*6,
        "fchannel": [1]*6,
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
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (5.0, {}))

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, cfg, **kwargs):
        captured["times"] = ts_dict.get("Po214", []).tolist()
        return FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    def fake_write(out_dir, summary, timestamp=None):
        if dataclasses.is_dataclass(summary):
            summary = dataclasses.asdict(summary)
        captured["summary"] = summary
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
        "--analysis-end-time", "1970-01-01T00:00:04Z",
        "--spike-end-time", "1970-01-01T00:00:00.5Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured.get("times") == [1.0, 3.0]
