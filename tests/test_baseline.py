import sys, json
from pathlib import Path
import pandas as pd
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import baseline_noise
from fitting import FitResult


def test_simple_baseline_subtraction(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_Po214": [7, 9],
            "hl_Po214": [1.0, 0.0],
            "eff_Po214": [1.0, 0.0],
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

    cal_mock = {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0), "peaks": {"Po210": {"centroid_adc": 10}}}
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
    rate = summary["baseline"]["rate_Bq"]["Po214"]
    assert rate == pytest.approx(0.2, rel=1e-3)
    assert summary["baseline"]["n_events"] == 2
    assert summary["baseline"]["dilution_factor"] == pytest.approx(1.0)
    assert summary["baseline"]["scales"]["Po214"] == pytest.approx(1.0)
    assert summary["baseline"]["scales"]["Po218"] == pytest.approx(1.0)
    assert summary["baseline"]["scales"]["Po210"] == pytest.approx(1.0)
    assert summary["baseline"]["scales"]["noise"] == pytest.approx(1.0)
    assert summary["time_fit"]["Po214"]["E_corrected"] == pytest.approx(0.8)
    assert summary["baseline"].get("noise_level") == 5.0
    times = list(captured.get("times", []))
    assert times == [20]
    # Ensure baseline events were not passed to the time fit
    assert all(t >= cfg["baseline"]["range"][1] for t in times)


def test_baseline_scaling_factor(tmp_path, monkeypatch):
    """Baseline subtraction scales by the monitor dilution factor."""
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 605.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_Po214": [7, 9],
            "hl_Po214": [1.0, 0.0],
            "eff_Po214": [1.0, 0.0],
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

    cal_mock = {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0), "peaks": {"Po210": {"centroid_adc": 10}}}
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
    rate = summary["baseline"]["rate_Bq"]["Po214"]
    dilution = summary["baseline"]["dilution_factor"]
    assert rate == pytest.approx(0.2, rel=1e-3)
    assert dilution == pytest.approx(0.5)
    assert summary["baseline"]["scales"]["Po214"] == pytest.approx(0.5)
    assert summary["baseline"]["scales"]["Po218"] == pytest.approx(0.5)
    assert summary["time_fit"]["Po214"]["E_corrected"] == pytest.approx(0.9)


def test_n0_prior_from_baseline(tmp_path, monkeypatch):
    """Baseline counts convert to an N0 prior in activity units."""
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_Po214": [7, 9],
            "hl_Po214": [1.0, 0.0],
            "eff_Po214": [1.0, 0.0],
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

    cal_mock = {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0), "peaks": {"Po210": {"centroid_adc": 10}}}
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
            "window_Po214": [7, 9],
            "hl_Po214": [1.0, 0.0],
            "eff_Po214": [1.0, 0.0],
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

    cal_mock = {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0), "peaks": {"Po210": {"centroid_adc": 10}}}
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
    assert summary["baseline"]["scales"]["Po214"] == pytest.approx(1.0)



def test_baseline_scaling_multiple_isotopes(tmp_path, monkeypatch):
    """Baseline subtraction scales multiple isotopes by dilution."""
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 100.0, "sample_volume_l": 50.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_Po214": [7, 9],
            "hl_Po214": [1.0, 0.0],
            "eff_Po214": [1.0, 0.0],
            "window_Po218": [5.5, 6.5],
            "hl_Po218": [1.0, 0.0],
            "eff_Po218": [1.0, 0.0],
            "window_Po210": [4.5, 5.5],
            "hl_Po210": [1.0, 0.0],
            "eff_Po210": [1.0, 0.0],
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
        "timestamp": [1, 2, 3, 20],
        "adc": [8, 6, 5, 8],
        "fchannel": [1, 1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cal_mock = {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0), "peaks": {"Po210": {"centroid_adc": 10}}}
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)

    def fake_fit_time_series(times_dict, t_start, t_end, cfg, **kwargs):
        return FitResult({
            "E_Po214": 0.5,
            "dE_Po214": 0.05,
            "E_Po218": 0.7,
            "dE_Po218": 0.07,
        }, np.zeros((2, 2)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit_time_series)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (1.0, {}, np.array([True, False, False])))

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
    base = summary["baseline"]
    dilution = 100.0 / 150.0
    assert base["dilution_factor"] == pytest.approx(dilution)
    assert base["rate_Bq"]["Po214"] == pytest.approx(0.1 * dilution, rel=1e-3)
    assert base["rate_Bq"]["Po218"] == pytest.approx(0.1 * dilution, rel=1e-3)
    assert base["rate_Bq"]["Po210"] == pytest.approx(0.1, rel=1e-3)
    assert base["rate_Bq"]["noise"] == pytest.approx(0.1, rel=1e-3)
    assert base["scales"] == {
        "Po214": dilution,
        "Po218": dilution,
        "Po210": 1.0,
        "noise": 1.0,
    }
    unc = 0.1 * dilution
    assert summary["time_fit"]["Po214"]["E_corrected"] == pytest.approx(0.5 - dilution * (0.1 * dilution), rel=1e-3)
    assert summary["time_fit"]["Po218"]["E_corrected"] == pytest.approx(0.7 - dilution * (0.1 * dilution), rel=1e-3)
    assert summary["time_fit"]["Po214"]["dE_corrected"] == pytest.approx(np.hypot(0.05, dilution * unc), rel=1e-3)
    assert summary["time_fit"]["Po218"]["dE_corrected"] == pytest.approx(np.hypot(0.07, dilution * unc), rel=1e-3)
