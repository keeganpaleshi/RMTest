import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from fitting import FitResult, FitParams


def test_analyze_systematics_runs(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [0, 10],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {"enable": True, "sigma_E_frac": 0.1},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [pd.Timestamp(1.0, unit="s", tz="UTC")],
        "adc": [5],
        "fchannel": [1],
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cal_mock = {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0), "peaks": {}}
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))

    monkeypatch.setattr(
        analyze,
        "fit_time_series",
        lambda *a, **k: FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0),
    )

    called = {}

    def fake_scan(fit_func, priors, sigmas):
        res = fit_func(priors)
        assert isinstance(res, dict)
        called["ok"] = True
        return {}, 0.0

    monkeypatch.setattr(analyze, "scan_systematics", fake_scan)
    monkeypatch.setattr(analyze, "write_summary", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    args = [
        "analyze.py",
        "--config", str(cfg_path),
        "--input", str(data_path),
        "--output_dir", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert called.get("ok") is True
