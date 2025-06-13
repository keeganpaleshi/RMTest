import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import math
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
