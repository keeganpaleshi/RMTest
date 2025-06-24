import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from calibration import CalibrationResult
from fitting import FitResult


def test_blue_weights_summary(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "efficiency": {
            "assay": [
                {"rate_cps": 1.0, "reference_bq": 10.0, "error": 0.1},
                {"rate_cps": 2.0, "reference_bq": 20.0, "error": 0.2},
            ]
        },
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({"fUniqueID": [1], "fBits": [0], "timestamp": [0], "adc": [1], "fchannel": [1]})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants",
        lambda *a, **k: CalibrationResult([0.0, 1.0], np.zeros((2, 2)), sigma_E=1.0),
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants_auto",
        lambda *a, **k: CalibrationResult([0.0, 1.0], np.zeros((2, 2)), sigma_E=1.0),
    )
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0, 0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    import efficiency

    monkeypatch.setattr(efficiency, "calc_spike_efficiency", lambda *a, **k: 0.1)
    monkeypatch.setattr(efficiency, "calc_assay_efficiency", lambda *a, **k: 0.05)
    monkeypatch.setattr(efficiency, "calc_decay_efficiency", lambda *a, **k: 0.1)

    def fake_blue(vals, errs):
        return 0.1, 0.01, np.array([0.8, 0.2])

    monkeypatch.setattr(efficiency, "blue_combine", fake_blue)

    captured = {}

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
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    weights = captured["summary"]["efficiency"].get("blue_weights")
    assert weights == [0.8, 0.2]
