import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def test_cli_radon_interval_overrides_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 5], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "analysis": {"radon_interval": ["0", "5"]},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [1.0],
        "adc": [8.0],
        "fchannel": [1],
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    captured = {}
    orig_load_config = analyze.load_config

    def fake_load_config(path):
        cfg_local = orig_load_config(path)
        captured["cfg"] = cfg_local
        return cfg_local

    monkeypatch.setattr(analyze, "load_config", fake_load_config)
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))

    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data_path),
        "--output_dir", str(tmp_path),
        "--radon-interval", "1970-01-01T00:00:01Z", "1970-01-01T00:00:02Z",
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    used = captured.get("cfg", {})
    assert used["analysis"]["radon_interval"] == [1.0, 2.0]
