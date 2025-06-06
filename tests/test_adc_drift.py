import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def _write_basic(tmp_path, drift_rate):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False, "adc_drift_rate": drift_rate},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [0.0, 1.0, 2.0],
        "adc": [10, 10, 10],
        "fchannel": [1, 1, 1],
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    return cfg_path, data_path


def test_adc_drift_applied(tmp_path, monkeypatch):
    cfg_path, data_path = _write_basic(tmp_path, 1.0)
    captured = {}

    def fake_shift(adc, ts, rate, t_ref=None):
        captured["shift_called"] = True
        captured["rate"] = rate
        return adc + 5

    def fake_cal(adc_vals, config=None):
        captured["cal_adc"] = np.array(adc_vals)
        return {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)}

    monkeypatch.setattr(analyze, "apply_linear_adc_shift", fake_shift)
    monkeypatch.setattr(analyze, "derive_calibration_constants", fake_cal)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", fake_cal)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: {})
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = summary
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured.get("shift_called") is True
    assert np.allclose(captured.get("cal_adc"), np.array([15, 15, 15]))
    assert captured["summary"].get("adc_drift_rate") == 1.0


def test_adc_drift_zero_noop(tmp_path, monkeypatch):
    cfg_path, data_path = _write_basic(tmp_path, 0.0)
    captured = {}

    def fake_shift(*a, **k):
        captured["called"] = True
        return a[0]

    def fake_cal(adc_vals, config=None):
        captured["cal_adc"] = np.array(adc_vals)
        return {"a": (1.0, 0.0), "c": (0.0, 0.0), "sigma_E": (1.0, 0.0)}

    monkeypatch.setattr(analyze, "apply_linear_adc_shift", fake_shift)
    monkeypatch.setattr(analyze, "derive_calibration_constants", fake_cal)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", fake_cal)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: {})
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = summary
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert captured.get("called") is None
    assert np.allclose(captured.get("cal_adc"), np.array([10, 10, 10]))
    assert captured["summary"].get("adc_drift_rate") == 0.0
