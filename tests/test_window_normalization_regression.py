"""Regression tests for window normalization in unbinned extended likelihood.

These tests ensure that:
1. EMG CDF is stable and computes correct in-window mass
2. Extended likelihood Poisson mean μ equals sum of yields after window renormalization
"""

import numpy as np
from src.rmtest.spectral.shapes import emg_pdf_E, emg_cdf_E
from src.rmtest.spectral.intensity import build_spectral_intensity, integral_of_intensity


def test_emg_window_mass_is_near_one_for_this_case():
    """Verify EMG CDF is stable and gives correct window mass.

    For a peak at mu=6.0 with sigma=0.05 and tau=0.10 in window [4.8, 8.3],
    essentially all the mass (>99.999%) should be in-window.
    """
    mu, sigma, tau = 6.0, 0.05, 0.10
    E_lo, E_hi = 4.8, 8.3

    pwin = float(emg_cdf_E(E_hi, mu, sigma, tau) - emg_cdf_E(E_lo, mu, sigma, tau))
    assert 0.99999 < pwin < 1.0, f"Expected pwin ≈ 1.0, got {pwin}"


def test_extended_mu_equals_sum_of_yields_after_window_renorm():
    """Verify extended likelihood μ equals sum of yields.

    After window renormalization, ∫[E_lo, E_hi] λ(E) dE should equal
    the sum of signal yields (300 + 400 + 300 = 1000 in this test).

    This is the critical fix: without window renormalization, μ would
    count mass outside the window, causing the optimizer to halve S_Po218.
    """
    E = np.linspace(4.8, 8.3, 6001)
    domain = (E.min(), E.max())
    iso_list = ["Po210", "Po218", "Po214"]

    # 300 + 400 + 300 yields, typical of synthetic test
    params = dict(
        N_Po210=300.0,
        mu_Po210=5.3,
        N_Po218=400.0,
        mu_Po218=6.0,
        tau_Po218=0.10,
        N_Po214=300.0,
        mu_Po214=7.7,
        sigma0=0.05,
        b0=0.0,
        b1=0.0,
    )

    use_emg = {"Po210": False, "Po218": True, "Po214": False}
    spectral = build_spectral_intensity(iso_list, use_emg, domain)
    lam = spectral(E, params)

    mu = integral_of_intensity(params, domain, iso_list=iso_list)

    # Unbinned extended rule: integral must equal total signal counts here
    expected_mu = 300 + 400 + 300
    assert abs(mu - expected_mu) < 1e-6, f"Expected μ = {expected_mu}, got {mu}"

    # Numerical integral of lam over the window should match too
    num_mu = np.trapz(lam, E)
    assert abs(num_mu - mu) / mu < 1e-3, (
        f"Numerical integral {num_mu} doesn't match analytical μ = {mu}"
    )
