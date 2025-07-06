import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import baseline_noise
from calibration import CalibrationResult
from dataclasses import asdict
from fitting import FitResult, FitParams


def test_baseline_noise_propagation(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"range": [0, 5], "monitor_volume_l": 605.0, "sample_volume_l": 1.0},
        "calibration": {"noise_cutoff": 8},
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
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [pd.Timestamp(1.0, unit="s", tz="UTC"), pd.Timestamp(2.0, unit="s", tz="UTC"), pd.Timestamp(20.0, unit="s", tz="UTC")],
        "adc": [10.0, 12.0, 50.0],
        "fchannel": [1, 1, 1],
    })
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

    captured = {}

    def fake_baseline_noise(adc, *args, **kwargs):
        captured["pedestal_cut"] = kwargs.get("pedestal_cut")
        return None, {}

    monkeypatch.setattr(baseline_noise, "estimate_baseline_noise", fake_baseline_noise)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1,1)), 0))

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
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured.get("pedestal_cut") == 8
