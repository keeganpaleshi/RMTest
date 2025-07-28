import json
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fitting import fit_spectrum, fit_time_series

from tests.synthetic_dataset import synthetic_spectrum, synthetic_dataset, PEAKS


def _priors():
    return {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (PEAKS["Po210"], 0.1),
        "S_Po210": (50, 5),
        "mu_Po218": (PEAKS["Po218"], 0.1),
        "S_Po218": (60, 6),
        "mu_Po214": (PEAKS["Po214"], 0.1),
        "S_Po214": (70, 7),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }


def test_resolution_freeze():
    energies = synthetic_spectrum()
    priors = _priors()
    res = fit_spectrum(energies, priors, flags={"fix_sigma0": True, "fix_F": True})
    assert res.params["sigma0"] == pytest.approx(priors["sigma0"][0])
    assert res.params["F"] == pytest.approx(priors["F"][0])


def test_amplitude_positive_refit():
    energies = synthetic_spectrum()
    priors = _priors()
    priors.update({"S_Po210": (-10, 5), "S_Po218": (-10, 5), "S_Po214": (-10, 5)})
    res = fit_spectrum(energies, priors)
    for key in ("S_Po210", "S_Po218", "S_Po214"):
        assert res.params[key] >= 0


def test_background_auto_selection():
    from background import estimate_polynomial_background_auto

    energies = synthetic_spectrum()
    coeffs, order = estimate_polynomial_background_auto(energies, PEAKS, max_order=2)
    assert order in (0, 1, 2)
    assert len(coeffs) == order + 1


def test_uncertainty_non_null():
    energies = synthetic_spectrum()
    priors = _priors()
    res = fit_spectrum(energies, priors)
    assert res.params.get("dS_Po210") is not None


def test_time_fit_validity_improved():
    times = np.arange(0, 50, 1.0)
    times_dict = {"Po214": times}
    cfg = {
        "isotopes": {"Po214": {"half_life_s": 1.0, "efficiency": 1.0}},
        "fit_background": True,
        "fit_initial": True,
    }
    res = fit_time_series(times_dict, 0.0, 50.0, cfg)
    assert res.params.get("fit_valid", False)


def _simulate_decay(rate, eff, T, rng):
    n = rng.poisson(rate * eff * T)
    return np.sort(rng.uniform(0, T, n))


@pytest.mark.parametrize("k", [2, 3])
def test_amplitude_scaling(k):
    T = 50.0
    eff = 0.8
    rng = np.random.default_rng(0)
    times = _simulate_decay(0.5, eff, T, rng)
    cfg = {
        "isotopes": {"Po214": {"half_life_s": 1.0, "efficiency": eff}},
        "fit_background": True,
        "fit_initial": True,
    }
    res1 = fit_time_series({"Po214": times}, 0.0, T, cfg)
    times_scaled = np.tile(times, k)
    res2 = fit_time_series({"Po214": times_scaled}, 0.0, T, cfg)
    ratio = res2.params["E_Po214"] / res1.params["E_Po214"]
    assert ratio == pytest.approx(k, rel=0.2)


def test_cli_smoke(tmp_path):
    df = synthetic_dataset()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {
            "method": "two-point",
            "peak_prominence": 0.0,
            "peak_width": 1,
            "nominal_adc": {"Po210": 5300, "Po218": 6000, "Po214": 7700},
            "peak_search_radius": 10,
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 5.0,
            "init_tau_adc": 1.0,
            "sanity_tolerance_mev": 1.0,
        },
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {
            "do_time_fit": True,
            "window_po214": [7.6, 7.8],
            "eff_po214": [1.0, 0.0],
            "hl_po214": [1.0],
            "flags": {},
        },
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run([
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "analyze.py"),
        "-i",
        str(csv),
        "-c",
        str(cfg_path),
        "-o",
        str(tmp_path),
    ], env=env)
    assert result.returncode == 0
    summary_file = next(tmp_path.glob("*/summary.json"))
    with open(summary_file) as f:
        summary = json.load(f)
    fit_valid = summary.get("time_fit", {}).get("Po214", {}).get("fit_valid")
    amp = summary.get("spectral_fit", {}).get("Po214", {}).get("S_Po214")
    unc = summary.get("time_fit", {}).get("Po214", {}).get("dE_Po214")
    assert fit_valid is True
    assert amp is None or amp >= 0
    assert unc is None or np.isfinite(unc)
