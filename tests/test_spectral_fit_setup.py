import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
from calibration import CalibrationResult
from fitting import FitParams, FitResult, fit_spectrum


def test_select_spectral_fit_frame_applies_fit_energy_range():
    df = pd.DataFrame(
        {
            "energy_MeV": [4.7, 4.8, 5.1, 8.3, 8.31],
            "adc": [4700, 4800, 5100, 8300, 8310],
        }
    )

    filtered, fit_range = analyze._select_spectral_fit_frame(
        df,
        {"fit_energy_range": [4.8, 8.3]},
    )

    assert fit_range == (4.8, 8.3)
    assert filtered["energy_MeV"].tolist() == [4.8, 5.1, 8.3]


def test_estimate_loglin_background_prior_counts_continuum():
    energies = np.array([4.9, 5.0, 5.29, 5.31, 5.99, 6.01, 7.0])
    mean, sigma = analyze._estimate_loglin_background_prior(
        energies,
        {"Po210": 5.3, "Po218": 6.0},
        peak_width=0.05,
        prior_hint=[0.0, 5.0],
    )

    assert mean == pytest.approx(3.0)
    assert sigma == pytest.approx(15.0)


def test_select_spectral_fit_frame_does_not_apply_peak_tolerance_padding():
    df = pd.DataFrame(
        {
            "energy_MeV": [4.55, 4.65, 4.8, 8.3, 8.35],
            "adc": [4550, 4650, 4800, 8300, 8350],
        }
    )

    filtered, fit_range = analyze._select_spectral_fit_frame(
        df,
        {"fit_energy_range": [4.8, 8.3], "spectral_peak_tolerance_mev": 0.2},
    )

    assert fit_range == pytest.approx((4.8, 8.3))
    np.testing.assert_allclose(filtered["energy_MeV"], [4.8, 8.3])


def test_fit_spectrum_loglin_unit_bootstrap_background_scale():
    rng = np.random.default_rng(0)
    energies = np.concatenate(
        [
            rng.normal(5.3, 0.05, 200),
            rng.normal(6.0, 0.05, 200),
            rng.normal(7.7, 0.05, 200),
            rng.uniform(4.8, 8.3, 400),
        ]
    )
    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (200, 20),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (200, 20),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (200, 20),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    res = fit_spectrum(energies, priors, flags={"background_model": "loglin_unit"})

    assert res.params["S_bkg"] > 100.0


def test_main_spectral_fit_uses_fit_range_and_background_prior(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "analysis": {"background_model": "loglin_unit"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": True,
            "spectral_binning_mode": "energy",
            "energy_bin_width": 0.5,
            "fit_energy_range": [4.8, 8.3],
            "expected_peaks": {"Po210": 5300, "Po218": 6000, "Po214": 7700},
            "mu_sigma": 0.05,
            "amp_prior_scale": 1.0,
            "spectral_peak_tolerance_mev": 0.05,
            "b0_prior": [0.0, 2.0],
            "b1_prior": [0.0, 2.0],
            "S_bkg_prior": [0.0, 5.0],
            "use_plot_bins_for_fit": True,
            "float_sigma_e": False,
            "flags": {"fix_f": True},
        },
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    df = pd.DataFrame(
        {
            "fUniqueID": np.arange(10),
            "fBits": np.zeros(10, dtype=int),
            "timestamp": pd.date_range("1970-01-01", periods=10, freq="s", tz="UTC"),
            "adc": [4700, 4800, 4900, 5300, 5310, 6000, 7700, 8200, 8300, 8400],
            "fchannel": np.ones(10, dtype=int),
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cal_result = CalibrationResult(
        coeffs=[0.0, 0.001],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=0.05,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_result)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_result)
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(analyze, "find_adc_bin_peaks", lambda *a, **k: {"Po210": 5300, "Po218": 6000, "Po214": 7700})
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    captured = {}
    plot_calls = []

    def fake_spectral_fit_with_check(energies, priors, flags, cfg, **kwargs):
        captured["energies"] = np.asarray(energies, dtype=float)
        captured["priors"] = dict(priors)
        captured["flags"] = dict(flags)
        captured["bin_edges"] = np.asarray(kwargs.get("bin_edges"), dtype=float)
        return FitResult(FitParams({"fit_valid": True}), np.zeros((0, 0)), 0), {}

    def fake_plot_spectrum(*, energies, fit_vals=None, out_png, bins=None, bin_edges=None, config=None, fit_flags=None, **kwargs):
        plot_calls.append(
            {
                "energies": np.asarray(energies, dtype=float),
                "bin_edges": None if bin_edges is None else np.asarray(bin_edges, dtype=float),
                "fit_flags": {} if fit_flags is None else dict(fit_flags),
            }
        )
        return None

    monkeypatch.setattr(analyze, "_spectral_fit_with_check", fake_spectral_fit_with_check)
    monkeypatch.setattr(analyze, "plot_spectrum", fake_plot_spectrum)
    monkeypatch.setattr(analyze, "write_summary", lambda out_dir, summary, timestamp=None: str(tmp_path / "out"))

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    energies = captured["energies"]
    assert np.all((energies >= 4.8) & (energies <= 8.3))
    np.testing.assert_allclose(energies, [4.8, 4.9, 5.3, 5.31, 6.0, 7.7, 8.2, 8.3])
    assert captured["priors"]["S_bkg"] == pytest.approx((4.0, 20.0))
    assert captured["flags"]["background_model"] == "loglin_unit"
    assert captured["bin_edges"][0] == pytest.approx(4.8)
    assert captured["bin_edges"][-1] == pytest.approx(8.3)
    assert len(plot_calls) == 2
    np.testing.assert_allclose(plot_calls[0]["energies"], [4.7, 4.8, 4.9, 5.3, 5.31, 6.0, 7.7, 8.2, 8.3, 8.4])
    assert plot_calls[0]["fit_flags"]["fit_energy_range"] == (4.8, 8.3)
    assert plot_calls[0]["bin_edges"][0] == pytest.approx(4.3)
    assert plot_calls[0]["bin_edges"][-1] == pytest.approx(8.8)
