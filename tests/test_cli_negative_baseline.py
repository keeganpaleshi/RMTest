import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import types
sys.modules.setdefault("pymc", types.ModuleType("pymc"))
from calibration import CalibrationResult
import analyze
import baseline_noise
from baseline_utils import BaselineError
import radon.baseline as rb
from dataclasses import asdict


def _common_setup(tmp_path, monkeypatch):
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
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)

    return cfg_path, data_path


def test_baseline_mode_none_skips_rate_subtraction(tmp_path, monkeypatch):
    cfg_path, data_path = _common_setup(tmp_path, monkeypatch)

    def fail(*a, **k):
        raise RuntimeError("subtraction called")

    monkeypatch.setattr(rb, "subtract_baseline_rate", fail)
    monkeypatch.setattr(analyze, "write_summary", lambda *a, **k: str((Path(a[0]) / (k.get("timestamp") or "x")).mkdir(parents=True, exist_ok=True)))
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data_path),
        "--output_dir", str(tmp_path),
        "--baseline_range", "1", "2",
        "--baseline-mode", "none",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()


def test_allow_negative_baseline_flag(tmp_path, monkeypatch):
    cfg_path, data_path = _common_setup(tmp_path, monkeypatch)

    def fake_subtract(*a, **k):
        return -0.1, 0.01, 0.2, 0.05

    monkeypatch.setattr(rb, "subtract_baseline_rate", fake_subtract)

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    def fake_summarize(cfg, iso):
        if not cfg.get("allow_negative_baseline"):
            raise BaselineError("neg")
        return {i: (0.1, 0.2, -0.1) for i in iso}

    monkeypatch.setattr(analyze, "summarize_baseline", fake_summarize)

    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data_path),
        "--output_dir", str(tmp_path),
        "--baseline_range", "1", "2",
        "--allow-negative-baseline",
        "--allow-negative-activity",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert "summary" in captured
