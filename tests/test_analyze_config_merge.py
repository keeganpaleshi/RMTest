import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import logging
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from calibration import CalibrationResult
from fitting import FitResult, FitParams
from dataclasses import asdict


def test_plot_time_series_receives_merged_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": False,
            "expected_peaks": {"Po210": 1250, "Po218": 1400, "Po214": 1800},
        },
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7.5, 8.0],
            "window_po218": None,
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {
            "plot_time_binning_mode": "fd",
            "plot_save_formats": ["png"],
            "overlay_isotopes": True,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(1000, unit="s", tz="UTC")],
            "adc": [7600],
            "fchannel": [1],
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    # Patch heavy functions with no-op versions
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants",
        lambda *a, **k: CalibrationResult(
            coeffs=[0.0, 1.0],
            cov=np.zeros((2, 2)),
            peaks={},
            sigma_E=1.0,
            sigma_E_error=0.0,
        ),
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants_auto",
        lambda *a, **k: CalibrationResult(
            coeffs=[0.0, 1.0],
            cov=np.zeros((2, 2)),
            peaks={},
            sigma_E=1.0,
            sigma_E_error=0.0,
        ),
    )
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)

    received = {}

    def fake_plot_time_series(*args, **kwargs):
        received.update(kwargs)
        Path(kwargs["out_png"]).touch()
        return None

    monkeypatch.setattr(analyze, "plot_time_series", fake_plot_time_series)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert received["config"]["plot_time_binning_mode"] == "fd"
    assert received["config"]["window_po214"] == [7.5, 8.0]
    assert received["config"]["overlay_isotopes"] is True


def test_legacy_background_model_key_reaches_summary_and_fit_flags(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": True,
            "expected_peaks": {"Po210": 1250, "Po218": 1400, "Po214": 1800},
            "bkg_mode": "loglin_unit",
            "mu_sigma": 0.05,
            "amp_prior_scale": 1.0,
            "b0_prior": [0.0, 2.0],
            "b1_prior": [0.0, 2.0],
            "S_bkg_prior": [0.0, 5.0],
        },
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3],
            "fBits": [0, 0, 0],
            "timestamp": [
                pd.Timestamp(1000, unit="s", tz="UTC"),
                pd.Timestamp(1001, unit="s", tz="UTC"),
                pd.Timestamp(1002, unit="s", tz="UTC"),
            ],
            "adc": [5300, 6000, 7700],
            "fchannel": [1, 1, 1],
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants",
        lambda *a, **k: CalibrationResult(
            coeffs=[0.0, 0.001],
            cov=np.zeros((2, 2)),
            peaks={},
            sigma_E=0.05,
            sigma_E_error=0.0,
        ),
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants_auto",
        lambda *a, **k: CalibrationResult(
            coeffs=[0.0, 0.001],
            cov=np.zeros((2, 2)),
            peaks={},
            sigma_E=0.05,
            sigma_E_error=0.0,
        ),
    )
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}

    def fake_spectral_fit_with_check(energies, priors, flags, cfg, **kwargs):
        captured["flags"] = dict(flags)
        return FitResult(FitParams({}), np.zeros((0, 0)), 0), {}

    monkeypatch.setattr(analyze, "_spectral_fit_with_check", fake_spectral_fit_with_check)

    saved = {}

    def fake_write(out_dir, summary, timestamp=None):
        saved["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    assert captured["flags"]["background_model"] == "loglin_unit"
    assert saved["summary"]["analysis"]["background_model"] == "loglin_unit"


def test_analysis_start_time_applied(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"analysis_start_time": "1970-01-01T00:00:10Z"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": False,
            "expected_peaks": {"Po210": 0, "Po218": 0, "Po214": 0},
        },
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7.5, 8.0],
            "window_po218": None,
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [pd.Timestamp(15, unit="s", tz="UTC")],
        "adc": [7600],
        "fchannel": [1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))

    captured = {}

    def fake_fit_time_series(times_dict, t_start, t_end, config, **kwargs):
        captured["t_start"] = t_start
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit_time_series)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "plot_radon_activity", lambda *a, **k: Path(a[2]).touch())
    monkeypatch.setattr(analyze, "plot_equivalent_air", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "plot_radon_trend", lambda *a, **k: Path(a[2]).touch(), raising=False)
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured.get("t_start") == 10.0


def test_job_id_overrides_results_folder(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": False,
            "expected_peaks": {"Po210": 1250, "Po218": 1400, "Po214": 1800},
        },
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7.5, 8.0],
            "window_po218": None,
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
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
        "adc": [1000],
        "fchannel": [1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    recorded = {}

    def fake_write_summary(out_dir, summary, timestamp=None):
        recorded["folder"] = Path(out_dir)
        recorded["folder"].mkdir(parents=True, exist_ok=True)
        return str(recorded["folder"])

    monkeypatch.setattr(analyze, "write_summary", fake_write_summary)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--job-id",
        "JOB123",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert recorded["folder"].name == "JOB123"


def test_efficiency_json_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7.5, 8.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "c.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)

    eff = {"spike": {"counts": 10, "activity_bq": 5, "live_time_s": 100}}
    eff_path = tmp_path / "eff.json"
    with open(eff_path, "w") as f:
        json.dump(eff, f)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    called = {}

    def fake_calc(counts, act, live):
        called["vals"] = (counts, act, live)
        return 0.1

    import efficiency
    monkeypatch.setattr(efficiency, "calc_spike_efficiency", fake_calc)
    monkeypatch.setattr(efficiency, "calc_assay_efficiency", lambda *a, **k: 0.1)
    monkeypatch.setattr(efficiency, "calc_decay_efficiency", lambda *a, **k: 0.1)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data),
        "--output-dir",
        str(tmp_path),
        "--efficiency-json",
        str(eff_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert called.get("vals") == (10, 5, 100)


def test_systematics_json_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": True, "window_po214": [7.4,7.9], "hl_po214": [1.0,0.0], "eff_po214": [1.0,0.0], "flags": {}},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "c2.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1800], "fchannel": [1]})
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)

    sys_cfg = {"enable": True}
    sys_path = tmp_path / "sys.json"
    with open(sys_path, "w") as f:
        json.dump(sys_cfg, f)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))

    monkeypatch.setattr(
        analyze,
        "fit_time_series",
        lambda *a, **k: FitResult(FitParams({"E": 0.0}), np.zeros((1, 1)), 0),
    )
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    called = {}
    def fake_scan(*a, **k):
        called["scan"] = True
        return ({}, 0.0)

    monkeypatch.setattr(analyze, "scan_systematics", fake_scan)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data),
        "--output-dir",
        str(tmp_path),
        "--systematics-json",
        str(sys_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert called.get("scan") is True


def test_time_bin_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7.5, 8.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)

    captured = {}

    def fake_plot_ts(*args, **kwargs):
        captured.update(kwargs)
        Path(kwargs["out_png"]).touch()
        return None

    monkeypatch.setattr(analyze, "plot_time_series", fake_plot_ts)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--plot-time-binning-mode",
        "fixed",
        "--plot-time-bin-width",
        "5",
        "--dump-ts-json",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["config"]["plot_time_binning_mode"] == "fixed"
    assert captured["config"]["plot_time_bin_width_s"] == 5.0
    assert captured["config"]["dump_time_series_json"] is True


def test_time_bin_override_logs(tmp_path, monkeypatch, caplog):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"], "plot_time_binning_mode": "auto"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--plot-time-binning-mode",
        "fd",
    ]
    monkeypatch.setattr(sys, "argv", args)
    with caplog.at_level(logging.INFO):
        analyze.main()

    assert "plotting.plot_time_binning_mode" in caplog.text


def test_po210_time_series_plot_generated(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7.5, 8.0],
            "window_po218": None,
            "window_po210": [5.2, 5.4],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "eff_po210": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)

    outputs = []

    def fake_plot(*args, **kwargs):
        outputs.append(Path(kwargs["out_png"]).name)
        Path(kwargs["out_png"]).touch()
        return None

    monkeypatch.setattr(analyze, "plot_time_series", fake_plot)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert "time_series_Po210.png" in outputs


def test_spike_count_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "efficiency": {"spike": {"activity_bq": 5, "live_time_s": 100}},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    import efficiency

    recorded = {}

    def fake_spike(cnt, act, live):
        recorded["cnt"] = cnt
        recorded["act"] = act
        recorded["live"] = live
        return 0.1

    monkeypatch.setattr(efficiency, "calc_spike_efficiency", fake_spike)

    saved = {}

    def fake_write(out_dir, summary, timestamp=None):
        saved["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
        "--spike-count",
        "42",
        "--spike-count-err",
        "2",
        "--spike-activity",
        "43",
        "--spike-duration",
        "44",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert recorded.get("cnt") == 42.0
    assert recorded.get("act") == 43.0
    assert recorded.get("live") == 44.0
    assert saved["summary"]["efficiency"]["sources"]["spike"]["error"] == 2.0


def test_spike_count_single_call(tmp_path, monkeypatch):
    """calc_spike_efficiency should only run once when --spike-count is used."""

    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "efficiency": {"spike": {"activity_bq": 5, "live_time_s": 100}},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    import efficiency

    calls = []

    def fake_spike(cnt, act, live):
        calls.append((cnt, act, live))
        return 0.1

    monkeypatch.setattr(efficiency, "calc_spike_efficiency", fake_spike)

    analyze._spike_eff_cache.clear()

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--spike-count",
        "42",
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    assert len(calls) == 1


def test_no_spike_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "efficiency": {"spike": {"counts": 10, "activity_bq": 5, "live_time_s": 100}},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    import efficiency

    calls = []

    def fake_spike(cnt, act, live):
        calls.append((cnt, act, live))
        return 0.1

    monkeypatch.setattr(efficiency, "calc_spike_efficiency", fake_spike)

    saved = {}

    def fake_write(out_dir, summary, timestamp=None):
        saved["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
        "--no-spike",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert calls == []
    assert "spike" not in saved["summary"]["efficiency"]["sources"]


def test_assay_efficiency_list(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "efficiency": {
            "assay": [
                {"rate_cps": 1.0, "reference_bq": 10.0, "error": 0.1},
                {"rate_cps": 2.0, "reference_bq": 20.0, "error": 0.2},
            ]
        },
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    import efficiency

    calls = []

    def fake_assay(rate, ref):
        calls.append((rate, ref))
        return 0.05

    recorded = {}

    def fake_blue(vals, errs):
        recorded["vals"] = list(vals)
        recorded["errs"] = list(errs)
        return 0.1, 0.01, np.array([0.5, 0.5])

    monkeypatch.setattr(efficiency, "calc_spike_efficiency", lambda *a, **k: 0.1)
    monkeypatch.setattr(efficiency, "calc_assay_efficiency", fake_assay)
    monkeypatch.setattr(efficiency, "calc_decay_efficiency", lambda *a, **k: 0.1)
    monkeypatch.setattr(efficiency, "blue_combine", fake_blue)

    saved = {}

    def fake_write(out_dir, summary, timestamp=None):
        saved["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert len(calls) == 2
    assert recorded["vals"] == [0.05, 0.05]
    assert recorded["errs"] == [0.1, 0.2]
    sources = saved["summary"]["efficiency"]["sources"]
    assert "assay_1" in sources and "assay_2" in sources


def test_spike_efficiency_list(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "efficiency": {
            "spike": [
                {"counts": 10, "activity_bq": 5, "live_time_s": 100, "error": 0.1},
                {"counts": 20, "activity_bq": 5, "live_time_s": 100, "error": 0.2},
            ]
        },
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    import efficiency

    calls = []

    def fake_spike(cnt, act, live):
        calls.append((cnt, act, live))
        return 0.05

    recorded = {}

    def fake_blue(vals, errs):
        recorded["vals"] = list(vals)
        recorded["errs"] = list(errs)
        return 0.1, 0.01, np.array([0.5, 0.5])

    monkeypatch.setattr(efficiency, "calc_spike_efficiency", fake_spike)
    monkeypatch.setattr(efficiency, "calc_assay_efficiency", lambda *a, **k: 0.1)
    monkeypatch.setattr(efficiency, "calc_decay_efficiency", lambda *a, **k: 0.1)
    monkeypatch.setattr(efficiency, "blue_combine", fake_blue)

    analyze._spike_eff_cache.clear()

    saved = {}

    def fake_write(out_dir, summary, timestamp=None):
        saved["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert len(calls) == 2
    assert recorded["vals"] == [0.05, 0.05]
    assert recorded["errs"] == [0.1, 0.2]
    sources = saved["summary"]["efficiency"]["sources"]
    assert "spike_1" in sources and "spike_2" in sources


def test_debug_flag_sets_log_level(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    captured = {}

    def fake_basic(level=None, **kwargs):
        captured["level"] = level

    monkeypatch.setattr(logging, "basicConfig", fake_basic)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--debug",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured.get("level") == logging.DEBUG


def test_settle_s_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2],
        "fBits": [0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(10.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0],
        "fchannel": [1, 1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict["Po214"].tolist()
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--settle-s",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["times"] == [10.0]


def test_settle_s_summary(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
        "--settle-s",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["summary"]["analysis"]["settle_s"] == 5.0


def test_analysis_end_time_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2],
        "fBits": [0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(10.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0],
        "fchannel": [1, 1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict["Po214"].tolist()
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--analysis-end-time",
        "1970-01-01T00:00:05Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["times"] == [0.0]


def test_analysis_start_time_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [pd.Timestamp(15.0, unit="s", tz="UTC")],
        "adc": [8.0],
        "fchannel": [1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["t_start"] = t_start
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--analysis-start-time",
        "1970-01-01T00:00:10Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["t_start"] == pytest.approx(15.0)


def test_spike_end_time_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2],
        "fBits": [0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(10.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0],
        "fchannel": [1, 1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict["Po214"].tolist()
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--spike-end-time",
        "1970-01-01T00:00:05Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["times"] == [10.0]


def test_spike_start_time_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2],
        "fBits": [0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(10.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0],
        "fchannel": [1, 1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict["Po214"].tolist()
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--spike-start-time",
        "1970-01-01T00:00:05Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["times"] == [0.0]


def test_spike_start_and_end_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [
            pd.Timestamp(0.0, unit="s", tz="UTC"),
            pd.Timestamp(2.0, unit="s", tz="UTC"),
            pd.Timestamp(6.0, unit="s", tz="UTC"),
        ],
        "adc": [8.0, 8.0, 8.0],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict["Po214"].tolist()
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--spike-start-time",
        "1970-01-01T00:00:01Z",
        "--spike-end-time",
        "1970-01-01T00:00:05Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["times"] == [0.0, 6.0]


def test_spike_period_cli(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC"), pd.Timestamp(6.0, unit="s", tz="UTC"), pd.Timestamp(12.0, unit="s", tz="UTC")],
        "adc": [8.0, 8.0, 8.0],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        captured["times"] = ts_dict["Po214"].tolist()
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    saved = {}

    def fake_write(out_dir, summary, timestamp=None):
        saved["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
        "--spike-period",
        "1970-01-01T00:00:00Z",
        "1970-01-01T00:00:05Z",
        "--spike-period",
        "1970-01-01T00:00:10Z",
        "1970-01-01T00:00:13Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["times"] == [6.0]
    exp = [
        [datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
         datetime(1970, 1, 1, 0, 0, 5, tzinfo=timezone.utc)],
        [datetime(1970, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
         datetime(1970, 1, 1, 0, 0, 13, tzinfo=timezone.utc)],
    ]
    assert saved["summary"]["analysis"]["spike_periods"] == exp


def test_seed_cli_sets_random_seed(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}
    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
        "--seed",
        "123",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["summary"]["random_seed"] == 123


def test_ambient_concentration_recorded(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"ambient_concentration": 0.5},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}
    def fake_plot_equivalent_air(t, v, e, conc, out_png, config=None):
        captured["conc"] = conc
        Path(out_png).touch()

    monkeypatch.setattr(analyze, "plot_equivalent_air", fake_plot_equivalent_air)

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
        "--ambient-concentration",
        "1.2",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["summary"]["analysis"]["ambient_concentration"] == 1.2
    assert captured["conc"] == 1.2


def test_ambient_concentration_from_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"ambient_concentration": 0.7},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}
    def fake_plot_equivalent_air(t, v, e, conc, out_png, config=None):
        captured["conc"] = conc
        Path(out_png).touch()

    monkeypatch.setattr(analyze, "plot_equivalent_air", fake_plot_equivalent_air)

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["summary"]["analysis"]["ambient_concentration"] == 0.7
    assert captured["conc"] == 0.7


def test_ambient_file_interpolation(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2],
        "fBits": [0, 0],
        "timestamp": [pd.Timestamp(0, unit="s", tz="UTC"), pd.Timestamp(2, unit="s", tz="UTC")],
        "adc": [1, 1],
        "fchannel": [1, 1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    amb = tmp_path / "amb.txt"
    np.savetxt(amb, [[0.0, 1.0], [2.0, 2.0]])

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    # Constant radon activity so volumes depend only on ambient interpolation
    import radon_activity
    monkeypatch.setattr(radon_activity, "compute_radon_activity", lambda *a, **k: (10.0, 1.0))

    captured = {}

    def fake_plot_equivalent_air(t, v, e, conc, out_png, config=None):
        captured["conc"] = conc
        captured["times"] = list(t)
        captured["vol"] = list(v)
        captured["err"] = list(e)
        Path(out_png).touch()

    monkeypatch.setattr(analyze, "plot_equivalent_air", fake_plot_equivalent_air)

    def fake_write(out_dir, summary, timestamp=None):
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
        "--ambient-file",
        str(amb),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured.get("conc") is None

    exp_times = np.linspace(0.0, 2.0, 100)
    amb = np.interp(exp_times, [0.0, 2.0], [1.0, 2.0])
    exp_vol = 10.0 / amb
    exp_err = 1.0 / amb
    assert np.allclose(captured["times"], exp_times.tolist())
    assert np.allclose(captured["vol"], exp_vol.tolist())
    assert np.allclose(captured["err"], exp_err.tolist())


def test_burst_mode_from_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
        "burst_filter": {"burst_mode": "none"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    recorded = {}

    def fake_burst(df, cfg, mode="rate"):
        recorded["mode"] = mode
        return df, 0

    monkeypatch.setattr(analyze, "apply_burst_filter", fake_burst)
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert recorded.get("mode") == "none"


def test_burst_mode_micro_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
        "burst_filter": {"burst_mode": "micro"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    recorded = {}

    def fake_burst(df, cfg, mode="rate"):
        recorded["mode"] = mode
        return df, 0

    monkeypatch.setattr(analyze, "apply_burst_filter", fake_burst)
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert recorded.get("mode") == "micro"


def test_burst_mode_cli_overrides(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
        "burst_filter": {"burst_mode": "none"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    recorded = {}

    def fake_burst(df, cfg, mode="rate"):
        recorded["mode"] = mode
        return df, 0

    monkeypatch.setattr(analyze, "apply_burst_filter", fake_burst)
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--burst-mode",
        "micro",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert recorded.get("mode") == "micro"


def test_burst_mode_summary_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    saved = {}

    def fake_write(out_dir, summary, timestamp=None):
        saved["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
        "--burst-mode",
        "micro",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert saved["summary"]["burst_filter"]["burst_mode"] == "micro"


def test_burst_filter_auto_disabled(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
        "burst_filter": {"burst_mode": "rate"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2],
            "fBits": [0, 0],
            "timestamp": [pd.Timestamp(0, unit="s", tz="UTC"), pd.Timestamp(2000, unit="s", tz="UTC")],
            "adc": [1, 1],
            "fchannel": [1, 1],
        }
    )
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    recorded = {}

    def fake_burst(df, cfg, mode="rate"):
        recorded["mode"] = mode
        return df, 0

    monkeypatch.setattr(analyze, "apply_burst_filter", fake_burst)
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants",
        lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0),
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants_auto",
        lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0),
    )
    monkeypatch.setattr(
        analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0, 0)), 0)
    )
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert recorded.get("mode") == "none"


def test_ambient_concentration_default_none(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["summary"]["analysis"]["ambient_concentration"] is None


def test_ambient_concentration_written_to_summary_file(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"ambient_concentration": 1.3},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "plot_equivalent_air", lambda *a, **k: Path(a[4]).touch())

    from io_utils import write_summary as real_write_summary

    results = {}

    def capture_write(out_dir, summary, timestamp=None):
        p = real_write_summary(out_dir, summary, timestamp)
        results["path"] = p
        return p

    monkeypatch.setattr(analyze, "write_summary", capture_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    summary_path = Path(results["path"]) / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["analysis"]["ambient_concentration"] == 1.3


def test_spike_periods_null_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"spike_periods": None},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }

    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC")], "adc": [8.0], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    saved = {}

    def fake_write(out_dir, summary, timestamp=None):
        saved["summary"] = asdict(summary)
        d = Path(out_dir) / "x"
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
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    assert saved["summary"]["analysis"]["spike_periods"] == []


def test_hl_po214_cli_overrides(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0.0, 20.0],
            "window_po218": [0.0, 20.0],
            "hl_po214": [1.0, 0.0],
            "hl_po218": [2.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "eff_po218": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }

    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC")], "adc": [8.0], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    calls = []

    def fake_fit(ts_dict, t_start, t_end, config, **kwargs):
        iso = list(ts_dict.keys())[0]
        calls.append((iso, config))
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
        "--hl-po214",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    values = {iso: c["isotopes"][iso]["half_life_s"] for iso, c in calls}
    assert values["Po214"] == 5.0



def test_hl_po210_default_used(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": False,
            "window_po210": [5.2, 5.4],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }

    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC")], "adc": [8.0], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))

    received = {}

    def fake_plot_time_series(*args, **kwargs):
        received.update(kwargs)
        Path(kwargs["out_png"]).touch()
        return None

    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0, 0)), 0))
    monkeypatch.setattr(analyze, "plot_time_series", fake_plot_time_series)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    assert "hl_po210" not in received["config"]


def test_time_fields_written_back(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": ["0", "1"], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "analysis": {
            "analysis_end_time": "1970-01-01T00:00:05Z",
            "spike_end_time": "1970-01-01T00:00:00Z",
            "spike_periods": [["1970-01-01T00:00:02Z", "1970-01-01T00:00:03Z"]],
            "run_periods": [["1970-01-01T00:00:00Z", "1970-01-01T00:00:10Z"]],
            "radon_interval": ["1970-01-01T00:00:03Z", "1970-01-01T00:00:05Z"],
        },
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }

    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [pd.Timestamp(0.5, unit="s", tz="UTC"), pd.Timestamp(2.5, unit="s", tz="UTC"), pd.Timestamp(4.5, unit="s", tz="UTC")],
        "adc": [8.0, 8.0, 8.0],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    captured = {}
    orig_load = analyze.load_config

    def fake_load(path):
        cfg_local = orig_load(path)
        return cfg_local

    monkeypatch.setattr(analyze, "load_config", fake_load)

    def fake_copy(outdir, cfg_in, exist_ok=False):
        captured["cfg"] = cfg_in

    monkeypatch.setattr(analyze, "copy_config", fake_copy)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: CalibrationResult(coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0))
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0, 0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(analyze, "scan_systematics", lambda *a, **k: ({}, 0.0))

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    used = captured.get("cfg", {})
    exp_end_time = datetime(1970, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    exp_spike_end = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert used["analysis"]["analysis_end_time"] == exp_end_time
    assert used["analysis"]["spike_end_time"] == exp_spike_end
    assert used["analysis"]["spike_periods"] == [
        [
            datetime(1970, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
        ]
    ]
    assert used["analysis"]["run_periods"] == [
        [
            datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(1970, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
        ]
    ]

    assert used["analysis"]["radon_interval"] == [
        datetime(1970, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
        datetime(1970, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
    ]
    exp_start = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    exp_end = datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

    assert used["baseline"]["range"] == [
        exp_start,
        exp_end,
    ]


def test_resolve_template_fixed_isotopes_defaults_to_auxiliary_only():
    fixed = analyze._resolve_template_fixed_isotopes(
        ["Po210", "Po216", "Bi212", "Unknown1"],
        {"fix_weak_isotopes": True},
    )

    assert fixed == {"Po216", "Bi212"}


def test_fit_time_bins_applies_centroid_controls(monkeypatch):
    import fitting

    timestamps = pd.date_range(
        "2024-01-01T00:00:00Z",
        periods=40,
        freq="1min",
        tz="UTC",
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_MeV": np.full(40, 7.69),
            "adc": np.arange(40, dtype=float),
        }
    )
    aggregate_result = FitResult(
        {
            "mu_Po214": 7.687,
            "sigma_Po214": 0.05,
            "S_Po214": 500.0,
            "S_bkg": 10.0,
            "sigma0": 0.10,
            "F": 0.0,
        },
        None,
        0,
    )

    captured = {}

    def fake_fit_spectrum(
        energies,
        priors,
        flags=None,
        bin_edges=None,
        bounds=None,
        pre_binned_hist=None,
        skip_minos=False,
        **kwargs,
    ):
        captured["bounds"] = bounds
        captured["penalty_priors"] = dict(flags.get("penalty_priors", {}))
        return {
            "fit_valid": True,
            "chi2_ndf": 4.2,
            "mu_Po214": 7.707,
            "S_Po214": 42.0,
            "dS_Po214": 3.0,
            "_n_bound_hits": 2,
            "_bound_hit_params": ["mu_Po214", "S_Po214"],
            "_bound_hits": {
                "mu_Po214": {"side": "upper"},
                "S_Po214": {"side": "upper"},
            },
            "_plot_edges": [7.5, 7.6, 7.7, 7.8],
            "_plot_hist": [1.0, 2.0, 1.0],
            "_plot_centers": [7.55, 7.65, 7.75],
            "_plot_model_total": [1.0, 2.0, 1.0],
            "_plot_components": {"Po214": [1.0, 2.0, 1.0]},
        }

    monkeypatch.setattr(fitting, "fit_spectrum", fake_fit_spectrum)

    cfg = {
        "spectral_fit": {"background_model": "none"},
        "time_fit": {
            "template_min_counts": 10,
            "float_centroids": True,
            "centroid_shift_bound_kev": 20.0,
            "centroid_shift_prior_sigma_kev": None,
        },
        "plotting": {
            "plot_time_bin_width_s": 3600,
            "plot_template_bin_fits": True,
            "plot_template_bin_fits_bad_only": True,
        },
    }
    spec_plot_data = {
        "flags": {"background_model": "none"},
        "bin_edges": np.array([7.5, 7.6, 7.7, 7.8], dtype=float),
    }

    result = analyze._fit_time_bins(
        df,
        aggregate_result,
        cfg,
        spec_plot_data=spec_plot_data,
    )

    lo, hi = captured["bounds"]["mu_Po214"]
    assert lo == pytest.approx(7.667)
    assert hi == pytest.approx(7.707)
    assert captured["penalty_priors"]["mu_Po214"][0] == pytest.approx(7.687)
    assert captured["penalty_priors"]["mu_Po214"][1] == pytest.approx(0.01)
    assert result["per_bin_diagnostics"][0]["shift_hit_limit"] is True
    assert result["per_bin_diagnostics"][0]["centroid_shift_kev"] == pytest.approx(20.0)
    assert result["per_bin_diagnostics"][0]["n_bound_hits"] == 2
    assert result["per_bin_diagnostics"][0]["bound_hit_params"] == ["mu_Po214", "S_Po214"]
    assert len(result["plot_entries"]) == 1
    assert result["plot_entries"][0]["n_bound_hits"] == 2


def test_should_store_template_bin_plot_when_non_centroid_bound_hit_present():
    assert analyze._should_store_template_bin_plot(
        {"fit_valid": True, "chi2_ndf": 1.2, "_n_bound_hits": 1},
        False,
        {
            "plot_template_bin_fits": True,
            "plot_template_bin_fits_bad_only": True,
            "plot_template_bin_fits_bad_chi2_ndf_min": 3.0,
        },
    )


def test_summarize_template_fit_results_counts_bound_hits():
    summary = analyze._summarize_template_fit_results(
        {
            "n_time_bins": 3,
            "n_fitted": 3,
            "n_valid": 2,
            "plot_entries": [{"bin_index": 0}],
            "centroid_control": {"limit_kev": 20.0},
            "per_bin_diagnostics": [
                {
                    "fit_valid": True,
                    "chi2_ndf": 2.0,
                    "centroid_shift_kev": 5.0,
                    "shift_hit_limit": False,
                    "n_bound_hits": 2,
                    "bound_hit_params": ["S_Po214", "mu_Po214"],
                },
                {
                    "fit_valid": True,
                    "chi2_ndf": 12.0,
                    "centroid_shift_kev": 10.0,
                    "shift_hit_limit": True,
                    "n_bound_hits": 1,
                    "bound_hit_params": ["S_Po214"],
                },
                {
                    "fit_valid": False,
                    "chi2_ndf": float("nan"),
                    "centroid_shift_kev": float("nan"),
                    "shift_hit_limit": False,
                    "n_bound_hits": 0,
                    "bound_hit_params": [],
                },
            ],
        }
    )

    assert summary["bins_with_any_bound_hits"] == 2
    assert summary["total_bound_hits"] == 3
    assert summary["bound_hit_param_counts"] == {"S_Po214": 2, "mu_Po214": 1}
    assert summary["non_centroid_bound_hit_param_counts"] == {"S_Po214": 2}
    assert summary["bins_with_non_centroid_bound_hits"] == 2
    assert summary["shift_hit_limit"] == 1
    assert summary["chi2_gt_10"] == 1


def test_write_template_bin_fit_plots_respects_log_toggle(tmp_path, monkeypatch):
    calls = {}

    def fake_plot_spectrum(*args, **kwargs):
        calls["config"] = dict(kwargs["config"])
        Path(kwargs["out_png"]).touch()

    monkeypatch.setattr(analyze, "plot_spectrum", fake_plot_spectrum)

    template_results = {
        "plot_entries": [
            {
                "bin_index": 0,
                "t": 0.0,
                "chi2_ndf": 5.0,
                "fit_valid": False,
                "shift_hit_limit": True,
                "fit_params": {
                    "_plot_edges": [7.5, 7.6, 7.7],
                    "_plot_hist": [1.0, 2.0],
                    "_plot_centers": [7.55, 7.65],
                    "_plot_model_total": [1.0, 2.0],
                    "_plot_components": {"Po214": [1.0, 2.0]},
                },
            }
        ]
    }

    manifest = analyze._write_template_bin_fit_plots(
        template_results,
        tmp_path,
        {
            "plotting": {
                "plot_template_bin_fits": True,
                "plot_template_bin_fits_log_scale": False,
            }
        },
    )

    assert len(manifest) == 1
    assert manifest[0]["png"].startswith("template_fit_bin_00000_")
    assert calls["config"]["plot_spectrum_write_log_copy"] is False
    assert (tmp_path / "template_bin_fits" / "template_fit_plot_index.json").exists()


def test_prefer_template_fit_prefers_non_clipped_retry():
    best = {"fit_valid": True, "chi2_ndf": 5.0}
    candidate = {"fit_valid": True, "chi2_ndf": 6.0}

    assert analyze._prefer_template_fit(
        best,
        candidate,
        best_hits_bound=True,
        candidate_hits_bound=False,
    )




