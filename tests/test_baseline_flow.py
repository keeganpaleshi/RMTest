import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import baseline_noise
from fitting import FitResult


def test_baseline_flow(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 10], "monitor_volume_l": 605.0, "sample_volume_l": 0.0},
        "calibration": {"noise_cutoff": 5},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": False
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [1, 2, 20],
        "adc": [2, 8, 8],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (1.0,0.0), "c": (0.0,0.0), "sigma_E": (1.0,0.0)})
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {}))

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = summary
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)

    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data_path),
        "--output_dir", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    summary = captured.get("summary", {})
    assert summary.get("baseline", {}).get("n_events") == 2
