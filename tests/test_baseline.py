import sys
import json
from pathlib import Path
import pandas as pd
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import baseline_noise
import baseline
from radon.baseline import subtract_baseline_counts
from fitting import FitResult
from calibration import CalibrationResult


def test_simple_baseline_subtraction(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
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
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [1, 2, 20],
        "adc": [8, 8, 8],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
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
    captured = {}

    def fake_fit_time_series(times_dict, t_start, t_end, cfg, **kwargs):
        captured["times"] = times_dict.get("Po214")
        return FitResult({"E_Po214": 1.0}, np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit_time_series)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (5.0, {}))

    def fake_write(out_dir, summary, timestamp=None):
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
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    summary = captured["summary"]
    # Baseline-subtracted rate is reported under ``corrected_rate_Bq``
    assert summary["baseline"]["n_events"] == 2
    assert summary["baseline"]["dilution_factor"] == pytest.approx(1.0)
    assert summary["baseline"]["scales"]["Po214"] == pytest.approx(1.0)
    assert summary["baseline"]["scales"]["Po218"] == pytest.approx(1.0)
    assert summary["baseline"]["scales"]["Po210"] == pytest.approx(1.0)
    assert summary["baseline"]["scales"]["noise"] == pytest.approx(1.0)
    corr_rate = summary["baseline"]["corrected_rate_Bq"]["Po214"]
    corr_sig = summary["baseline"]["corrected_sigma_Bq"]["Po214"]

    eff = cfg["time_fit"]["eff_po214"][0]
    live_time = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    counts = summary["baseline"]["analysis_counts"]["Po214"]
    base_counts = summary["baseline"]["rate_Bq"]["Po214"] * summary["baseline"]["live_time"] * eff
    exp_rate, exp_sigma = subtract_baseline_counts(
        counts, eff, live_time, base_counts, summary["baseline"]["live_time"]
    )

    assert corr_rate == pytest.approx(exp_rate)
    assert corr_sig == pytest.approx(exp_sigma)
    assert summary["baseline"].get("noise_level") == 5.0
    times = list(captured.get("times", []))
    assert times == [1, 2, 20]


def test_baseline_scaling_factor(tmp_path, monkeypatch):
    """Baseline subtraction scales by the monitor dilution factor."""
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 605.0},
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
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [1, 2, 20],
        "adc": [8, 8, 8],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
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
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({"E_Po214": 1.0}, np.zeros((1,1)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
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
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    summary = captured["summary"]
    dilution = summary["baseline"]["dilution_factor"]
    assert dilution == pytest.approx(0.5)
    assert summary["baseline"]["scales"]["Po214"] == pytest.approx(0.5)
    assert summary["baseline"]["scales"]["Po218"] == pytest.approx(0.5)
    corr_rate = summary["baseline"]["corrected_rate_Bq"]["Po214"]
    corr_sig = summary["baseline"]["corrected_sigma_Bq"]["Po214"]

    eff = cfg["time_fit"]["eff_po214"][0]
    live_time = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    counts = summary["baseline"]["analysis_counts"]["Po214"]
    base_counts = summary["baseline"]["rate_Bq"]["Po214"] * summary["baseline"]["live_time"] * eff
    exp_rate, exp_sigma = subtract_baseline_counts(
        counts, eff, live_time, base_counts, summary["baseline"]["live_time"]
    )

    assert corr_rate == pytest.approx(exp_rate)
    assert corr_sig == pytest.approx(exp_sigma)


def test_n0_prior_from_baseline(tmp_path, monkeypatch):
    """Baseline counts convert to an N0 prior in activity units."""
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7, 9],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": True},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [1, 2, 20],
        "adc": [8, 8, 8],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
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

    captured = {}

    def fake_fit_time_series(times_dict, t_start, t_end, cfg, **kwargs):
        return FitResult({"E_Po214": 1.0}, np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit_time_series)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    def fake_scan_systematics(fit_func, priors, sigma_dict):
        captured["priors"] = priors
        try:
            fit_func(priors)
        except Exception:
            pass
        return {}, 0.0

    monkeypatch.setattr(analyze, "scan_systematics", fake_scan_systematics)

    def fake_write(out_dir, summary, timestamp=None):
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

    n0_prior = captured.get("priors", {}).get("N0", (None,))[0]
    assert n0_prior == pytest.approx(0.2, rel=1e-3)
    assert captured["summary"]["baseline"]["scales"]["Po214"] == pytest.approx(1.0)


def test_isotopes_to_subtract_control(tmp_path, monkeypatch):
    """No subtraction occurs when the isotope list is empty."""
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 0.0, "isotopes_to_subtract": []},
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
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [1, 2, 20],
        "adc": [8, 8, 8],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
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
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({"E_Po214": 1.0}, np.zeros((1,1)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
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
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    summary = captured["summary"]
    assert "rate_Bq" not in summary.get("baseline", {})
    assert "E_corrected" not in summary["time_fit"]["Po214"]
    assert "dE_corrected" not in summary["time_fit"]["Po214"]
    assert summary["baseline"]["scales"]["Po214"] == pytest.approx(1.0)


def test_baseline_scaling_multiple_isotopes(tmp_path, monkeypatch):
    """Baseline subtraction scales each isotope by the dilution factor."""
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {
            "range": [0, 10],
            "monitor_volume_l": 605.0,
            "sample_volume_l": 200.0,
        },
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7, 9],
            "window_po218": [5.8, 6.3],
            "window_po210": [5.2, 5.4],
            "hl_po214": [1.0, 0.0],
            "hl_po218": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "eff_po218": [1.0, 0.0],
            "eff_po210": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3, 4, 5, 6],
            "fBits": [0] * 6,
            "timestamp": [1, 2, 3, 4, 20, 21],
            "adc": [8.0, 6.0, 5.3, 2.0, 8.0, 6.0],
            "fchannel": [1] * 6,
        }
    )
    data_path = tmp_path / "data.csv"
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

    fit_params = {
        "E_Po214": 1.0,
        "dE_Po214": 0.05,
        "E_Po218": 2.0,
        "dE_Po218": 0.1,
    }
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(fit_params, np.zeros((2, 2)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    mask_noise = np.array([0, 0, 0, 1], dtype=bool)

    def fake_noise(adc, *args, **kw):
        return 5.0, {}, mask_noise

    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", fake_noise)

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
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

    summary = captured["summary"]
    dilution = summary["baseline"]["dilution_factor"]
    noise_rate = summary["baseline"]["rate_Bq"]["noise"]

    assert noise_rate == pytest.approx(0.1, rel=1e-3)

    assert summary["baseline"]["scales"] == {
        "Po214": pytest.approx(dilution),
        "Po218": pytest.approx(dilution),
        "Po210": pytest.approx(1.0),
        "noise": pytest.approx(1.0),
    }

    assert "E_corrected" in summary["time_fit"]["Po214"]
    assert "dE_corrected" in summary["time_fit"]["Po214"]

    corr_rate = summary["baseline"]["corrected_rate_Bq"]["Po214"]
    corr_sig = summary["baseline"]["corrected_sigma_Bq"]["Po214"]

    eff = cfg["time_fit"]["eff_po214"][0]
    live_time = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    counts = summary["baseline"]["analysis_counts"]["Po214"]
    base_counts = summary["baseline"]["rate_Bq"]["Po214"] * summary["baseline"]["live_time"] * eff
    exp_rate, exp_sigma = subtract_baseline_counts(
        counts, eff, live_time, base_counts, summary["baseline"]["live_time"]
    )

    assert corr_rate == pytest.approx(exp_rate)
    assert corr_sig == pytest.approx(exp_sigma)
    # Po-218 fit results may be absent in this minimal dataset


def test_noise_level_none_not_recorded(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
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
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [1, 2, 20],
        "adc": [8, 8, 8],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
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
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({"E_Po214": 1.0}, np.zeros((1,1)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
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
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    summary = captured.get("summary", {})
    assert "noise_level" not in summary.get("baseline", {})


def test_sigma_rate_uses_weighted_counts(tmp_path, monkeypatch):
    """Baseline-corrected uncertainty should use weighted iso counts."""
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
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
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [1, 2, 20],
        "adc": [8, 8, 8],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
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
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({"E_Po214": 1.0}, np.zeros((1,1)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    baseline_e = np.array([8.0, 8.0])
    def fake_window_prob(E, sigma, lo, hi):
        arr = np.asarray(E)
        if arr.size == 2 and np.allclose(arr, baseline_e):
            return np.full_like(arr, 2.0)
        return np.ones_like(arr)

    monkeypatch.setattr(analyze, "window_prob", fake_window_prob)

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
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
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    dE_corr = captured["summary"]["time_fit"]["Po214"]["dE_corrected"]
    assert dE_corr == pytest.approx(0.2198, rel=1e-3)


def test_rate_histogram_single_event():
    df = pd.DataFrame({"timestamp": [1.0], "adc": [10.0]})
    bins = np.array([0, 20])
    rate, live = baseline.rate_histogram(df, bins)
    assert live == 0.0
    assert np.all(rate == 0.0)
