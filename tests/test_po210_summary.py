import sys
from pathlib import Path
import json
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import radon_activity
from fitting import FitResult


def test_po210_time_fit_written(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_Po210": [5.0, 6.0],
            "hl_Po210": [1.0, 0.0],
            "eff_Po210": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0.0], "adc": [5.5], "fchannel": [1]})
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})

    import numpy as np

    def fake_fit(ts, start, end, cfg, weights=None):
        assert "Po210" in ts
        return FitResult({"E_Po210": 2.0, "dE_Po210": 0.1}, np.zeros((1, 1)), 0)
    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    calls = []
    def fake_radon(*args, **kwargs):
        calls.append(args)
        return (1.0, 0.1)
    monkeypatch.setattr(radon_activity, "compute_radon_activity", fake_radon)

    captured = {}
    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = summary
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)
    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    args = ["analyze.py", "--config", str(cfg_path), "--input", str(data_path), "--output_dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    summary = captured.get("summary", {})
    assert summary["time_fit"]["Po210"]["E_Po210"] == pytest.approx(2.0)
    assert calls and calls[0][6] == 2.0
