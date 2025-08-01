import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import math
import json
import pandas as pd
import analyze
from calibration import CalibrationResult
from fitting import FitResult, FitParams
from systematics import scan_systematics, apply_linear_adc_shift


def test_scan_systematics_with_dict_result():
    # Priors for two parameters
    priors = {
        "p1": (1.0, 0.1),
        "p2": (2.0, 0.2),
    }

    sigma_dict = {"p1": 0.5, "p2": 0.25}

    def dummy_fit(p):
        # Simple deterministic function returning parameter means plus 1
        return {k: v[0] + 1 for k, v in p.items()}

    deltas, total_unc = scan_systematics(dummy_fit, priors, sigma_dict)
    assert deltas["p1"] == pytest.approx(0.5)
    assert deltas["p2"] == pytest.approx(0.25)
    # Total uncertainty should be sqrt(0.5^2 + 0.25^2)
    expected_unc = (0.5**2 + 0.25**2) ** 0.5
    assert total_unc == pytest.approx(expected_unc)


def test_apply_linear_adc_shift_noop():
    adc = np.array([100, 101, 102])
    t = np.array([0.0, 1.0, 2.0])
    out = apply_linear_adc_shift(adc, t, 0.0)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, adc)


def test_apply_linear_adc_shift_rate():
    adc = np.zeros(3)
    t = np.array([0.0, 1.0, 2.0])
    out = apply_linear_adc_shift(adc, t, 1.0)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [0.0, 1.0, 2.0])


def test_apply_linear_adc_shift_quadratic():
    adc = np.zeros(3)
    t = np.array([0.0, 1.0, 2.0])
    out = apply_linear_adc_shift(
        adc,
        t,
        0.0,
        mode="quadratic",
        params={"a": 0.5, "b": 1.0},
    )
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [0.0, 1.5, 4.0])


def test_apply_linear_adc_shift_piecewise():
    adc = np.zeros(4)
    t = np.array([0.0, 1.0, 2.0, 3.0])
    params = {"times": [0.0, 2.0, 3.0], "shifts": [0.0, 1.0, 1.5]}
    out = apply_linear_adc_shift(
        adc,
        t,
        0.0,
        mode="piecewise",
        params=params,
    )
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [0.0, 0.5, 1.0, 1.5])


def test_scan_systematics_with_adc_drift():
    adc = np.zeros(3)
    t = np.array([0.0, 1.0, 2.0])

    def fit_func(p):
        slope = p["drift"][0]
        shifted = apply_linear_adc_shift(adc, t, slope)
        return np.mean(shifted)

    priors = {"drift": (0.0, 0.0)}
    sig = {"drift": 1.0}
    deltas, tot = scan_systematics(fit_func, priors, sig)
    assert deltas["drift"] == pytest.approx(1.0)
    assert tot == pytest.approx(1.0)


def test_scan_systematics_fractional_and_absolute():
    priors = {"sigma_E": (2.0, 0.1), "mu": (5.0, 0.1)}

    def fit_func(p):
        return {k: v[0] for k, v in p.items()}

    shifts = {"sigma_E_frac": 0.1, "mu_keV": 2.0}
    deltas, tot = scan_systematics(fit_func, priors, shifts)
    assert deltas["sigma_E"] == pytest.approx(0.2)
    assert deltas["mu"] == pytest.approx(0.002)
    expected = math.sqrt(0.2 ** 2 + 0.002 ** 2)
    assert tot == pytest.approx(expected)


def test_analyze_systematics_skip_unknown(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7.0, 8.0],
            "hl_po214": [1.0, 0.0],
            "eff_po214": [1.0, 0.0],
            "flags": {},
        },
        "systematics": {
            "enable": True,
            "sigma_E_frac": 0.1,
            "tail_fraction": 0.2,
            "energy_shift_keV": 1.0,
        },
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [pd.Timestamp(0.0, unit="s", tz="UTC")],
        "adc": [10],
        "fchannel": [1],
    })
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)

    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants",
        lambda *a, **k: cal_mock,
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants_auto",
        lambda *a, **k: cal_mock,
    )

    monkeypatch.setattr(
        analyze,
        "fit_time_series",
        lambda *a, **k: FitResult(FitParams({"E_Po214": 1.0}), np.zeros((1, 1)), 0),
    )
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())

    captured = {}

    def fake_scan(fit_func, priors, sigma_dict):
        captured["sigma"] = dict(sigma_dict)
        return {}, 0.0

    monkeypatch.setattr(analyze, "scan_systematics", fake_scan)

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

    assert captured.get("sigma") == {}


def test_scan_systematics_repeatable():
    priors = {"x": (1.0, 0.1)}

    def fit_func(p):
        return {"x": p["x"][0] + 1.0}

    shifts = {"x": 0.5}

    d1, unc1 = scan_systematics(fit_func, priors, shifts)
    d2, unc2 = scan_systematics(fit_func, priors, shifts)

    assert d1 == d2
    assert unc1 == unc2
