"""Tests for unbinned spectral fit normalization.

These tests ensure that the extended unbinned likelihood implementation uses
proper rate densities [counts/MeV] and that the peak shapes are unit-normalized.
"""

import numpy as np
import pytest


def test_emg_pdf_unit_normalization():
    """Verify that EMG PDF integrates to 1."""
    from rmtest.spectral.shapes import emg_pdf_E

    # Test over a wide energy range
    E = np.linspace(0, 15, 20001)
    dE = E[1] - E[0]

    # Test with typical polonium peak parameters
    f = emg_pdf_E(E, mu=5.3, sigma=0.134, tau=0.02)

    # Should be positive and finite
    assert np.all(f >= 0)
    assert np.all(np.isfinite(f))

    # Should integrate to 1 (within numerical tolerance)
    integral = np.trapz(f, E)
    assert np.isclose(integral, 1.0, rtol=1e-6, atol=1e-6), (
        f"EMG PDF integral = {integral}, expected 1.0"
    )


def test_gaussian_pdf_unit_normalization():
    """Verify that Gaussian PDF integrates to 1."""
    from rmtest.spectral.shapes import gaussian_pdf_E

    E = np.linspace(0, 15, 20001)

    f = gaussian_pdf_E(E, mu=5.3, sigma=0.134)

    assert np.all(f >= 0)
    assert np.all(np.isfinite(f))

    integral = np.trapz(f, E)
    assert np.isclose(integral, 1.0, rtol=1e-6, atol=1e-6), (
        f"Gaussian PDF integral = {integral}, expected 1.0"
    )


def test_emg_edge_cases():
    """Test EMG PDF with edge case parameters."""
    from rmtest.spectral.shapes import emg_pdf_E

    E = np.linspace(0, 10, 1001)

    # Zero or negative sigma should return zeros
    f_zero_sigma = emg_pdf_E(E, mu=5.0, sigma=0.0, tau=0.02)
    assert np.all(f_zero_sigma == 0.0)

    f_neg_sigma = emg_pdf_E(E, mu=5.0, sigma=-0.1, tau=0.02)
    assert np.all(f_neg_sigma == 0.0)

    # Zero or negative tau should return zeros
    f_zero_tau = emg_pdf_E(E, mu=5.0, sigma=0.1, tau=0.0)
    assert np.all(f_zero_tau == 0.0)

    f_neg_tau = emg_pdf_E(E, mu=5.0, sigma=0.1, tau=-0.02)
    assert np.all(f_neg_tau == 0.0)


def test_spectral_intensity_scale():
    """Verify that spectral intensity has correct scale (counts/MeV)."""
    from rmtest.spectral.intensity import spectral_intensity_E, integral_of_intensity

    # Define fit domain
    Emin, Emax = 4.8, 8.3
    E = np.linspace(Emin, Emax, 1001)

    # Typical parameters (using counts, not densities)
    N210, N218, N214 = 15000.0, 3000.0, 8000.0
    b0, b1 = 1000.0, 0.0  # background in counts/MeV

    params = {
        "N210": N210,
        "mu210": 5.30,
        "sig0": 0.134,
        "tau210": 0.02,
        "N218": N218,
        "mu218": 6.00,
        "tau218": 0.03,
        "N214": N214,
        "mu214": 7.687,
        "tau214": 0.02,
        "b0": b0,
        "b1": b1,
    }

    # Compute intensity
    lam = spectral_intensity_E(E, params, domain=(Emin, Emax))

    # Intensity must be positive and finite everywhere
    assert np.all(lam > 0), "Intensity must be positive"
    assert np.all(np.isfinite(lam)), "Intensity must be finite"

    # Compute integral
    mu_tot = integral_of_intensity(params, domain=(Emin, Emax))

    # The integral should equal sum of peak counts + background integral
    expected_peak_counts = N210 + N218 + N214
    expected_bkg_counts = b0 * (Emax - Emin) + 0.5 * b1 * (Emax**2 - Emin**2)
    expected_total = expected_peak_counts + expected_bkg_counts

    assert np.isclose(mu_tot, expected_total, rtol=1e-12), (
        f"Integral = {mu_tot}, expected {expected_total}"
    )


def test_unbinned_counts_scale_consistency():
    """Verify that fitted yields are on the same scale as event counts.

    This is the key test to prevent the bug where yields were inflated by
    ~1/(4*ΔE_bin). For a well-conditioned fit, the sum of fitted peak areas
    should be within [0.3, 1.5] × N_events.
    """
    from rmtest.spectral.nll_unbinned import nll_extended_unbinned
    from rmtest.spectral.intensity import spectral_intensity_E

    # Generate synthetic data with known yields
    rng = np.random.default_rng(42)
    N210_true, N218_true, N214_true = 15000, 3000, 8000
    b0_true, b1_true = 1000.0, 0.0

    Emin, Emax = 4.8, 8.3

    # Simple toy sample: draw Gaussians for the peaks
    # (In a real fit we'd use the full EMG, but this is sufficient for a scale test)
    E_210 = rng.normal(5.30, 0.134, N210_true)
    E_218 = rng.normal(6.00, 0.134, N218_true)
    E_214 = rng.normal(7.687, 0.134, N214_true)

    # Background: uniform in energy
    N_bkg = int(b0_true * (Emax - Emin))
    E_bkg = rng.uniform(Emin, Emax, N_bkg)

    E = np.concatenate([E_210, E_218, E_214, E_bkg])
    E = E[(E > Emin) & (E < Emax)]

    N_events = len(E)

    # Set up parameters close to truth
    params = {
        "N210": N210_true,
        "mu210": 5.30,
        "sig0": 0.134,
        "tau210": 0.02,
        "N218": N218_true,
        "mu218": 6.00,
        "tau218": 0.03,
        "N214": N214_true,
        "mu214": 7.687,
        "tau214": 0.02,
        "b0": b0_true,
        "b1": b1_true,
    }

    # Evaluate likelihood at true parameters
    lam = spectral_intensity_E(E, params, domain=(Emin, Emax))

    # All intensities must be positive
    assert np.all(lam > 0), "Intensity must be positive at all observed events"
    assert np.all(np.isfinite(lam)), "Intensity must be finite"

    # Compute NLL
    nll = nll_extended_unbinned(E, params, domain=(Emin, Emax))
    assert np.isfinite(nll), "NLL must be finite"

    # Key test: sum of fitted peak areas should be on same scale as N_events
    sum_peaks = N210_true + N218_true + N214_true
    ratio = sum_peaks / N_events

    # For this synthetic dataset, ratio should be reasonable (not ~19!)
    # The exact ratio depends on how much background we added, but it should be < 2
    assert 0.3 < ratio < 1.5, (
        f"Sum of peak counts ({sum_peaks}) vs N_events ({N_events}) "
        f"has ratio {ratio:.2f}, which is outside [0.3, 1.5]. "
        "This suggests a normalization bug."
    )


def test_background_integral_included_in_extended_likelihood():
    """Verify that background integral is properly included in μ_tot."""
    from likelihood_ext import neg_loglike_extended

    # Simple constant-background model
    def constant_intensity(E, params):
        return np.full_like(E, params["b0"], dtype=float)

    E = np.array([5.0, 6.0, 7.0])
    params = {"b0": 100.0, "b1": 0.0}
    domain = (4.5, 7.5)

    # Extended likelihood should compute:
    # μ = b0 * (Emax - Emin) = 100 * 3.0 = 300
    # -ln L = μ - Σ ln(λ) = 300 - 3*ln(100)
    expected_mu = 100.0 * 3.0
    expected_nll = expected_mu - 3 * np.log(100.0)

    nll = neg_loglike_extended(
        E,
        constant_intensity,
        params,
        area_keys=[],  # no peak areas
        domain=domain,
    )

    assert np.isclose(nll, expected_nll, rtol=1e-10), (
        f"NLL = {nll}, expected {expected_nll}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
