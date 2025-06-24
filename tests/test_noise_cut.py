import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import numpy as np
from calibration import CalibrationResult


def test_noise_cutoff_cli_overrides(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {"noise_cutoff": 100},

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


    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2],
            "fBits": [0, 0],
            "timestamp": [1.0, 2.0],
            "adc": [5, 15],
            "fchannel": [1, 1],
        }
    )


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
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))

    captured = {}

    def fake_fit(ts_dict, t_start, t_end, cfg, weights=None, **kwargs):
        captured["times"] = ts_dict.get("Po214", []).tolist()
        return {"E_Po214": 1.0}

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
        "--noise-cutoff", "5",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured.get("times") == [2.0]
    assert captured["summary"]["noise_cut"]["removed_events"] == 1
