import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from calibration import CalibrationResult
from datetime import datetime, timezone
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from dataclasses import asdict
import baseline_noise
import baseline_handling
from fitting import FitResult, FitParams


def test_cli_baseline_range_overrides_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 5], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
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
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3],
            "fBits": [0, 0, 0],
            "timestamp": [pd.Timestamp(0.5, unit="s", tz="UTC"), pd.Timestamp(1.5, unit="s", tz="UTC"), pd.Timestamp(2.5, unit="s", tz="UTC")],
            "adc": [8.0, 8.0, 8.0],
            "fchannel": [1, 1, 1],
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
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    orig_fixed_background = baseline_handling.get_fixed_background_for_time_fit

    def fake_fixed_background(record, isotope, config):
        result = orig_fixed_background(record, isotope, config)
        if result:
            result = dict(result)
            result["mode"] = "baselineFixed"
        return result

    monkeypatch.setattr(
        baseline_handling,
        "get_fixed_background_for_time_fit",
        fake_fixed_background,
    )

    captured = {}

    orig_load_config = analyze.load_config

    def fake_load_config(path):
        cfg_local = orig_load_config(path)
        captured["cfg"] = cfg_local
        return cfg_local

    monkeypatch.setattr(analyze, "load_config", fake_load_config)

    def fake_fit(ts_dict, t_start, t_end, cfg, **kwargs):
        captured["times"] = ts_dict.get("Po214", []).tolist()
        return FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

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
        "--baseline_range", "1", "2",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    summary = captured.get("summary", {})
    exp_start = datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    exp_end = datetime(1970, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    assert summary.get("baseline", {}).get("start") == exp_start
    assert summary.get("baseline", {}).get("end") == exp_end
    assert summary.get("baseline", {}).get("n_events") == 1
    assert captured.get("cfg", {}).get("baseline", {}).get("range") == [
        exp_start,
        exp_end,
    ]
    tf_summary = summary.get("time_fit", {}).get("Po214", {})
    assert tf_summary.get("background_mode") == "fixed_from_baseline"
    assert tf_summary.get("baseline_rate_Bq") == pytest.approx(1.0)
    assert tf_summary.get("background_source") == "fixed_from_baseline"

    radon_summary = summary.get("radon", {})
    plot_payload = radon_summary.get("plot_series", {})
    if plot_payload:
        assert plot_payload.get("background_mode") == "fixed_from_baseline"


def test_time_fit_background_mode_floated(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
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
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2],
            "fBits": [0, 0],
            "timestamp": [
                pd.Timestamp(0.5, unit="s", tz="UTC"),
                pd.Timestamp(1.5, unit="s", tz="UTC"),
            ],
            "adc": [8.0, 8.0],
            "fchannel": [1, 1],
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
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    activity_calls: list[str | None] = []
    total_calls: list[str | None] = []

    def fake_plot_radon_activity(*args, **kwargs):
        out_png = Path(args[2])
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.touch()
        activity_calls.append(kwargs.get("background_mode"))
        return None

    def fake_plot_total_radon(*args, **kwargs):
        out_png = Path(args[2])
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.touch()
        total_calls.append(kwargs.get("background_mode"))
        return None

    monkeypatch.setattr(analyze, "plot_radon_activity", fake_plot_radon_activity)
    monkeypatch.setattr(analyze, "plot_total_radon", fake_plot_total_radon)
    monkeypatch.setattr(
        analyze,
        "plot_radon_trend",
        lambda *a, **k: Path(a[2]).touch(),
        raising=False,
    )

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, cfg_fit, **kwargs):
        return FitResult(
            FitParams({"E_Po214": 1.0, "B_Po214": 0.2}),
            np.zeros((2, 2)),
            0,
        )

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

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

    tf_summary = captured.get("summary", {}).get("time_fit", {}).get("Po214", {})
    assert tf_summary.get("background_mode") == "floated"
    assert "baseline_rate_Bq" not in tf_summary

    expected_mode = tf_summary.get("background_mode")
    assert expected_mode == "floated"
    radon_plot_series = (
        captured.get("summary", {}).get("radon", {}).get("plot_series", {})
    )
    if radon_plot_series:
        assert radon_plot_series.get("background_mode") == expected_mode
    assert activity_calls
    assert total_calls
    assert activity_calls[-1] == expected_mode
    assert total_calls[-1] == expected_mode
