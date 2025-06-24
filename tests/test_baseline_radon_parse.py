import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from fitting import FitResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def test_baseline_and_radon_intervals_parsed(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 1], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "analysis": {},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0.5], "adc": [1], "fchannel": [1]})
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)})
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0, 0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}
    monkeypatch.setattr(analyze, "copy_config", lambda outdir, cfg_in, exist_ok=False: captured.update({"cfg": cfg_in}))

    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data),
        "--output_dir", str(tmp_path),
        "--baseline_range", "1970-01-01T00:00:01Z", "1970-01-01T00:00:02Z",
        "--radon-interval", "1970-01-01T00:00:03Z", "1970-01-01T00:00:04Z",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    used = captured.get("cfg", {})
    assert used["baseline"]["range"] == ["1970-01-01T00:00:01+00:00", "1970-01-01T00:00:02+00:00"]
    assert used["analysis"]["radon_interval"] == ["1970-01-01T00:00:03+00:00", "1970-01-01T00:00:04+00:00"]
