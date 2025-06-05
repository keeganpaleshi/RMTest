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
