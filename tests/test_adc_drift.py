import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from fitting import FitResult
from calibration import CalibrationResult


def _write_basic(tmp_path, drift_rate, mode="linear", params=None):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {
            "enable": False,
            "adc_drift_rate": drift_rate,
            "adc_drift_mode": mode,
            "adc_drift_params": params,
        },
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

    def fake_shift(adc, ts, rate, t_ref=None, mode="linear", params=None):
        captured["shift_called"] = True
        captured["rate"] = rate
        captured["mode"] = mode
        captured["params"] = params
        return adc + 5

    def fake_cal(adc_vals, config=None):
        captured["cal_adc"] = np.array(adc_vals)
        return CalibrationResult(
            coeffs=[0.0, 1.0],
            cov=np.zeros((2, 2)),
            peaks={},
            sigma_E=1.0,
            sigma_E_error=0.0,
        )

    monkeypatch.setattr(analyze, "apply_linear_adc_shift", fake_shift)
    monkeypatch.setattr(analyze, "derive_calibration_constants", fake_cal)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", fake_cal)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0,0)), 0))
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
    assert isinstance(captured.get("cal_adc"), np.ndarray)
    assert np.allclose(captured.get("cal_adc"), np.array([15, 15, 15]))
    assert captured["summary"].get("adc_drift_rate") == 1.0
    assert captured["mode"] == "linear"
    assert captured["params"] is None


def test_adc_drift_zero_noop(tmp_path, monkeypatch):
    cfg_path, data_path = _write_basic(tmp_path, 0.0)
    captured = {}

    def fake_shift(*a, **k):
        captured["called"] = True
        return a[0]

    def fake_cal(adc_vals, config=None):
        captured["cal_adc"] = np.array(adc_vals)
        return CalibrationResult(
            coeffs=[0.0, 1.0],
            cov=np.zeros((2, 2)),
            peaks={},
            sigma_E=1.0,
            sigma_E_error=0.0,
        )

    monkeypatch.setattr(analyze, "apply_linear_adc_shift", fake_shift)
    monkeypatch.setattr(analyze, "derive_calibration_constants", fake_cal)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", fake_cal)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0,0)), 0))
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
    assert isinstance(captured.get("cal_adc"), np.ndarray)
    assert np.allclose(captured.get("cal_adc"), np.array([10, 10, 10]))
    assert captured["summary"].get("adc_drift_rate") == 0.0


def test_adc_drift_quadratic_cfg(tmp_path, monkeypatch):
    cfg_path, data_path = _write_basic(
        tmp_path, 0.0, mode="quadratic", params={"a": 0.5, "b": 1.0}
    )
    captured = {}

    def fake_shift(adc, ts, rate, t_ref=None, mode="linear", params=None):
        captured["mode"] = mode
        captured["params"] = params
        return adc

    monkeypatch.setattr(analyze, "apply_linear_adc_shift", fake_shift)
    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "write_summary", lambda *a, **k: Path(a[0]).mkdir(exist_ok=True) or str(a[0]))
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

    assert captured.get("mode") == "quadratic"
    assert captured.get("params") == {"a": 0.5, "b": 1.0}


def test_adc_drift_piecewise_cfg(tmp_path, monkeypatch):
    cfg_path, data_path = _write_basic(
        tmp_path,
        0.0,
        mode="piecewise",
        params={"times": [0.0, 1.0], "shifts": [0.0, 1.0]},
    )
    captured = {}

    def fake_shift(adc, ts, rate, t_ref=None, mode="linear", params=None):
        captured["mode"] = mode
        captured["params"] = params
        return adc

    monkeypatch.setattr(analyze, "apply_linear_adc_shift", fake_shift)
    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "write_summary", lambda *a, **k: Path(a[0]).mkdir(exist_ok=True) or str(a[0]))
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

    assert captured.get("mode") == "piecewise"
    assert captured.get("params") == {"times": [0.0, 1.0], "shifts": [0.0, 1.0]}

def test_adc_drift_warning_on_failure(tmp_path, monkeypatch, capsys):
    """ADC drift correction failure should emit a warning but not abort."""
    cfg_path, data_path = _write_basic(tmp_path, 1.0)

    def bad_shift(*a, **k):
        raise ValueError("boom")

    def fake_cal(adc_vals, config=None):
        return CalibrationResult(
            coeffs=[0.0, 1.0],
            cov=np.zeros((2, 2)),
            peaks={},
            sigma_E=1.0,
            sigma_E_error=0.0,
        )

    monkeypatch.setattr(analyze, "apply_linear_adc_shift", bad_shift)
    monkeypatch.setattr(analyze, "derive_calibration_constants", fake_cal)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", fake_cal)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0,0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())

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
    out = capsys.readouterr().out

    assert "Could not apply ADC drift correction" in out
    assert captured["summary"].get("adc_drift_rate") == 1.0
