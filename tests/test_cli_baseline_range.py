import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import baseline_noise
from fitting import FitResult


def test_cli_baseline_range_overrides_config(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 5], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_Po214": [0, 20],
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

    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3],
            "fBits": [0, 0, 0],
            "timestamp": [0.5, 1.5, 2.5],
            "adc": [8.0, 8.0, 8.0],
            "fchannel": [1, 1, 1],
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cal_mock = {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0), "peaks": {"Po210": {"centroid_adc": 10}}}
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    captured = {}

    orig_load_config = analyze.load_config

    def fake_load_config(path):
        cfg_local = orig_load_config(path)
        captured["cfg"] = cfg_local
        return cfg_local

    monkeypatch.setattr(analyze, "load_config", fake_load_config)

    def fake_fit(ts_dict, t_start, t_end, cfg):
        captured["times"] = ts_dict.get("Po214", []).tolist()
        return FitResult({"E_Po214": 1.0}, np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

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
        "--baseline_range", "1", "2",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    summary = captured.get("summary", {})
    assert summary.get("baseline", {}).get("start") == 1.0
    assert summary.get("baseline", {}).get("end") == 2.0
    assert summary.get("baseline", {}).get("n_events") == 1
    assert captured.get("cfg", {}).get("baseline", {}).get("range") == ["1", "2"]
