import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


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
            "window_Po214": [7.5, 8.0],
            "window_Po218": None,
            "hl_Po214": [1.0, 0.0],
            "eff_Po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {
            "plot_time_binning_mode": "fd",
            "plot_save_formats": ["png"],
            "overlay_isotopes": True,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [1000],
            "adc": [7600],
            "fchannel": [1],
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    # Patch heavy functions with no-op versions
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: {})
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
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert received["config"]["plot_time_binning_mode"] == "fd"
    assert received["config"]["window_Po214"] == [7.5, 8.0]
    assert received["config"]["overlay_isotopes"] is True


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
            "window_Po214": [7.5, 8.0],
            "window_Po218": None,
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
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [15],
        "adc": [7600],
        "fchannel": [1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})

    captured = {}

    def fake_fit_time_series(times_dict, t_start, t_end, config):
        captured["t_start"] = t_start
        return {}

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit_time_series)
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
        "--output_dir",
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
            "window_Po214": [7.5, 8.0],
            "window_Po218": None,
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
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [0],
        "adc": [1000],
        "fchannel": [1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: {})
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    recorded = {}

    def fake_write_summary(out_dir, summary, timestamp=None):
        recorded["folder"] = Path(out_dir) / timestamp
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
        "--output_dir",
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
            "window_Po214": [7.5, 8.0],
            "hl_Po214": [1.0, 0.0],
            "eff_Po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "c.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0], "adc": [1], "fchannel": [1]})
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)

    eff = {"spike": {"counts": 10, "activity_bq": 5, "live_time_s": 100}}
    eff_path = tmp_path / "eff.json"
    with open(eff_path, "w") as f:
        json.dump(eff, f)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: {})
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
        "--output_dir",
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
        "time_fit": {"do_time_fit": True, "window_Po214": [7.4,7.9], "hl_Po214": [1.0,0.0], "eff_Po214": [1.0,0.0], "flags": {}},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "c2.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0], "adc": [1800], "fchannel": [1]})
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)

    sys_cfg = {"enable": True, "sigma_shifts": {}, "scan_keys": []}
    sys_path = tmp_path / "sys.json"
    with open(sys_path, "w") as f:
        json.dump(sys_cfg, f)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})

    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: {"E":0.0})
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
        "--output_dir",
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
            "window_Po214": [7.5, 8.0],
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

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: {})
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
        "--output_dir",
        str(tmp_path),
        "--time-bin-mode",
        "fixed",
        "--time-bin-width",
        "5",
        "--dump-ts-json",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured["config"]["plot_time_binning_mode"] == "fixed"
    assert captured["config"]["plot_time_bin_width_s"] == 5.0
    assert captured["config"]["dump_time_series_json"] is True
