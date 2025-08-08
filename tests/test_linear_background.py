import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from background import (
    estimate_linear_background,
    estimate_polynomial_background_auto,
)
import analyze
from calibration import CalibrationResult
from fitting import FitResult, FitParams


def generate_spectrum():
    rng = np.random.default_rng(0)
    peaks = {
        "Po210": 5.3,
        "Po218": 6.0,
        "Po214": 7.7,
    }
    energies = np.concatenate([
        rng.normal(peaks["Po210"], 0.05, 150),
        rng.normal(peaks["Po218"], 0.05, 150),
        rng.normal(peaks["Po214"], 0.05, 150),
    ])
    # Linear continuum: b0=50, b1=2
    e_bg = np.linspace(5.0, 8.0, 60)
    counts = (50 + 2 * e_bg).astype(int)
    cont = np.concatenate([np.full(c, e) for e, c in zip(e_bg, counts)])
    energies = np.concatenate([energies, cont])
    return energies, peaks


def generate_poly_spectrum(bg_func):
    rng = np.random.default_rng(1)
    peaks = {"Po210": 5.3, "Po218": 6.0, "Po214": 7.7}
    energies = np.concatenate(
        [
            rng.normal(peaks["Po210"], 0.05, 150),
            rng.normal(peaks["Po218"], 0.05, 150),
            rng.normal(peaks["Po214"], 0.05, 150),
        ]
    )
    e_bg = np.linspace(5.0, 8.0, 60)
    counts = bg_func(e_bg).astype(int)
    cont = np.concatenate([np.full(c, e) for e, c in zip(e_bg, counts)])
    energies = np.concatenate([energies, cont])
    return energies, peaks


def test_estimate_linear_background():
    energies, peaks = generate_spectrum()
    b0, b1 = estimate_linear_background(energies, peaks, peak_width=0.3)
    assert b0 != 0
    assert b1 != 0


def test_linear_background_slope_unbiased():
    """Estimated slope should match the true continuum slope."""
    energies, peaks = generate_spectrum()
    b0, b1 = estimate_linear_background(energies, peaks, peak_width=0.3)
    assert b0 == pytest.approx(200.0, rel=0.05)
    assert b1 == pytest.approx(8.0, rel=0.05)


def test_auto_background_priors(monkeypatch, tmp_path):
    energies, peaks = generate_spectrum()
    rng = np.random.default_rng(1)
    adc = (energies * 1000).astype(int)
    df = pd.DataFrame({
        "fUniqueID": np.arange(len(adc)),
        "fBits": np.zeros(len(adc)),
        "timestamp": rng.uniform(0, 1, len(adc)),
        "adc": adc,
        "fchannel": np.ones(len(adc)),
    })
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)

    peaks_adc = {k: int(v * 1000) for k, v in peaks.items()}
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": True,
            "bkg_mode": "auto",
            "expected_peaks": peaks_adc,
            "mu_sigma": 0.05,
            "amp_prior_scale": 1.0,
            "spectral_binning_mode": "fd",
            "peak_search_width_adc": 1,
            "mu_bounds": {"Po210": [5.2, 5.4], "Po218": [5.9, 6.1], "Po214": [7.6, 7.8]},
        },
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "c.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    captured = {}

    def fake_fit_spectrum(energies, priors, **kw):
        captured.update(priors)
        return FitResult(FitParams({}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_spectrum", fake_fit_spectrum)
    cal_mock = CalibrationResult(
        coeffs=[0.0, 0.001],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=0.05,
        sigma_E_error=0.01,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "write_summary", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0,0)), 0))

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(csv),
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    b0_man, b1_man = estimate_linear_background(energies, peaks, peak_width=0.3)
    e_lo = float(min(energies))
    e_hi = float(max(energies))
    B_est = b0_man * (e_hi - e_lo) + 0.5 * b1_man * (e_hi**2 - e_lo**2)
    beta0_est = np.log(max(b0_man, 1e-12))
    beta1_est = b1_man / max(b0_man, 1e-12)
    assert captured["S_bkg"][0] == pytest.approx(B_est, rel=0.05)
    assert captured["beta0"][0] == pytest.approx(beta0_est, rel=0.05)
    assert captured["beta1"][0] == pytest.approx(beta1_est, rel=0.1)


def test_zero_count_bins():
    """estimate_linear_background should handle empty histogram bins."""
    rng = np.random.default_rng(42)
    peaks = {"Po210": 5.3, "Po214": 7.7}
    energies = np.concatenate([
        rng.normal(peaks["Po210"], 0.01, 50),
        rng.normal(peaks["Po214"], 0.01, 50),
    ])

    b0, b1 = estimate_linear_background(energies, peaks, peak_width=0.3, bins=50)
    assert not np.isnan(b0)
    assert not np.isnan(b1)


def test_polynomial_background_order_selection():
    flat = lambda e: np.full_like(e, 30)
    slope = lambda e: 20 + 5 * (e - 6)
    quad = lambda e: 10 + 2 * (e - 6) ** 2

    for func, expected in [(flat, 0), (slope, 1), (quad, 2)]:
        energies, peaks = generate_poly_spectrum(func)
        coeffs, order = estimate_polynomial_background_auto(
            energies, peaks, max_order=2
        )
        assert order == expected
