import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import math
from systematics import scan_systematics, apply_linear_adc_shift
from fitting import fit_spectrum, FitResult


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
    assert deltas["mu"] == pytest.approx(2.0)
    expected = math.sqrt(0.2 ** 2 + 2.0 ** 2)
    assert tot == pytest.approx(expected)


def test_scan_systematics_on_fitresult():
    rng = np.random.default_rng(0)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 150),
        rng.normal(6.0, 0.05, 150),
        rng.normal(7.7, 0.05, 150),
    ])

    priors = {
        "sigma_E": (0.05, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (150, 15),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (150, 15),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (150, 15),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    base = fit_spectrum(energies, priors, bins=30)
    scan_priors = {
        k: (base.params[k], base.params.get("d" + k, 0.0))
        for k in base.params
        if not k.startswith("d") and k != "fit_valid"
    }
    scan_priors["energy_shift"] = (0.0, 1.0)
    scan_priors["tail_fraction"] = (0.0, 1.0)

    def wrapper(p):
        shift = p.get("energy_shift", (0.0, 0.0))[0]
        pri = {k: v for k, v in p.items() if k not in ("energy_shift", "tail_fraction")}
        for iso in ("Po210", "Po218", "Po214"):
            key = f"mu_{iso}"
            if key in pri:
                mu, sig = pri[key]
                pri[key] = (mu + shift / 1000.0, sig)
        out = fit_spectrum(energies, pri, bins=30)
        out.params["energy_shift"] = shift
        out.params["tail_fraction"] = p.get("tail_fraction", (0.0,))[0]
        return out

    deltas, _ = scan_systematics(wrapper, scan_priors, {"sigma_E_frac": 0.2})
    assert deltas["sigma_E"] != 0.0

    d2, _ = scan_systematics(wrapper, scan_priors, {"energy_shift_keV": 1.0})
    assert d2["energy_shift"] == pytest.approx(1.0)
