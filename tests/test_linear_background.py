import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from background import estimate_linear_background
import analyze
from fitting import FitResult


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


def test_estimate_linear_background():
    energies, peaks = generate_spectrum()
    b0, b1 = estimate_linear_background(energies, peaks, peak_width=0.3)
    assert b0 != 0
    assert b1 != 0


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
        return FitResult({}, np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_spectrum", fake_fit_spectrum)
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: {"a": (0.001, 0.0), "c": (0.0, 0.0), "sigma_E": (0.05, 0.01)})
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: {"a": (0.001, 0.0), "c": (0.0, 0.0), "sigma_E": (0.05, 0.01)})
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "write_summary", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult({}, np.zeros((0,0)), 0))

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
    assert captured["b0"][0] == pytest.approx(b0_man, rel=0.05)
    assert captured["b1"][0] == pytest.approx(b1_man, rel=0.1)
