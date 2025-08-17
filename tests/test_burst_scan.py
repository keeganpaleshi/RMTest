import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import CalibrationResult
import analyze
from fitting import FitResult, FitParams


def test_burst_sensitivity_scan(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po218": [5.9, 6.1],
            "window_po214": [7.5, 7.8],
            "hl_po218": [1.0],
            "hl_po214": [1.0],
            "eff_po218": [1.0],
            "eff_po214": [1.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
        "burst_filter": {"burst_mode": "rate"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    times = np.linspace(0, 100, 50)
    burst = np.full(20, 30)
    all_times = np.concatenate([times, burst])
    df = pd.DataFrame({
        "fUniqueID": range(len(all_times)),
        "fBits": [0]*len(all_times),
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in all_times],
        "adc": [6.0]*len(all_times),
        "fchannel": [1]*len(all_times),
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "plot_spectrum_comparison", lambda *a, **k: {})
    monkeypatch.setattr(analyze, "plot_activity_grid", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "_eff_prior", lambda v: (1.0, 0.0))

    def fake_fit(ts_dict, t_start, t_end, cfg, **kwargs):
        params = {f"E_{k}": len(v)/(t_end - t_start) for k, v in ts_dict.items()}
        return FitResult(FitParams(params), np.zeros((len(params), len(params))), 0, counts=sum(len(v) for v in ts_dict.values()))

    monkeypatch.setattr(analyze, "fit_time_series", fake_fit)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
        "--burst-sensitivity-scan",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    summary_file = next(Path(tmp_path).glob("*/summary.json"))
    with open(summary_file) as f:
        summary = json.load(f)

    assert summary["burst_filter"].get("sensitivity_scan")
    assert summary["burst_filter"]["sensitivity_scan"]["grid"]
