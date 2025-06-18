import sys
from pathlib import Path
import json
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import radon_activity
import numpy as np
from fitting import FitResult


def test_total_radon_uses_sample_volume(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"monitor_volume_l": 10.0, "sample_volume_l": 5.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": True, "window_po214": [7, 9], "hl_po214": 1.0, "eff_Po214": [1.0, 0.0], "flags": {}},
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
        "adc": [8],
        "fchannel": [1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    cal_mock = {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0), "peaks": {"Po210": {"centroid_adc": 10}}}
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    monkeypatch.setattr(radon_activity, "compute_radon_activity", lambda *a, **k: (5.0, 0.5))

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
    expected = radon_activity.compute_total_radon(5.0, 0.5, 10.0, 5.0)[2]
    assert summary["radon_results"]["total_radon_in_sample_Bq"]["value"] == pytest.approx(expected)
