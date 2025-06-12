import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from fitting import FitResult


def test_hierarchical_summary(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
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
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    # Existing run summaries
    for i in range(2):
        d = tmp_path / f"old{i}"
        d.mkdir()
        with open(d / "summary.json", "w") as f:
            json.dump({
                "half_life": 10.0 + i,
                "dhalf_life": 1.0,
                "calibration": {"a": [1.0, 0.1], "c": [0.0, 0.1]},
            }, f)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, None, 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}

    def fake_fit(runs, draws=2000, tune=1000, chains=2, random_seed=42):
        captured["runs"] = runs
        return {"half_life": {"mean": 11.0}}

    monkeypatch.setattr(analyze, "fit_hierarchical_runs", fake_fit)

    def fake_write(out_dir, summary, timestamp=None):
        d = Path(out_dir) / "new"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "summary.json", "w") as f:
            json.dump({
                "half_life": 12.0,
                "dhalf_life": 1.0,
                "calibration": {"a": [1.2, 0.1], "c": [0.2, 0.1]},
            }, f)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    out_json = tmp_path / "hier.json"
    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data_path),
        "--output_dir", str(tmp_path),
        "--hierarchical-summary", str(out_json),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    assert len(captured.get("runs", [])) == 3
    assert out_json.exists()
    with open(out_json) as f:
        res = json.load(f)
    assert res["half_life"]["mean"] == 11.0
