import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze


def test_pipeline_calibrates_after_fix(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {
            "method": "two-point",
            "peak_prominence": 0.0,
            "peak_width": 1,
            "nominal_adc": {"Po210": 1250, "Po218": 1300, "Po214": 1800},
            "peak_search_radius": 200,
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 5.0,
            "init_tau_adc": 1.0,
            "sanity_tolerance_mev": 1.0,
        },
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    data_path = Path(__file__).resolve().parents[1] / "example_input.csv"

    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = summary
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)

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
    cent = summary.get("calibration", {}).get("peaks", {}).get("Po218", {}).get("centroid_adc")
    assert cent is not None
    assert cent == pytest.approx(1300, abs=3)

