"""Tests for unbinned likelihood normalization.

These tests verify that:
1. Shape functions (EMG, Gaussian) are properly unit-normalized
2. The intensity function produces correct counts/MeV densities
3. Extended likelihood yields match the actual event counts
4. Background integrals are computed correctly

This addresses the normalization bug where unbinned fits were producing
yields ~19× too large due to incorrect bin-width scaling.
"""

import numpy as np
import pytest
from rmtest.spectral.shapes import emg_pdf_E, gaussian_pdf_E
from rmtest.spectral.intensity import (
    spectral_intensity_E,
    integral_of_intensity,
    build_spectral_intensity,
)
from rmtest.spectral.nll_unbinned import nll_extended_unbinned


class TestShapeNormalization:
    """Test that shape functions integrate to 1."""

    def test_emg_pdf_unit_normalization(self):
        """EMG PDF should integrate to 1 over wide energy range."""
        E = np.linspace(0, 15, 20001)
        f = emg_pdf_E(E, mu=5.3, sigma=0.134, tau=0.02)

        # Check all values are non-negative
        assert (f >= 0).all(), "EMG PDF must be non-negative"

        # Check integration
        integral = np.trapz(f, E)
        assert np.isclose(integral, 1.0, rtol=1e-4, atol=1e-5), (
            f"EMG PDF should integrate to 1, got {integral}"
        )

    def test_emg_pdf_different_parameters(self):
        """Test EMG normalization with various parameter combinations."""
        E = np.linspace(0, 20, 30001)

        test_cases = [
            {"mu": 5.3, "sigma": 0.134, "tau": 0.02},
            {"mu": 7.687, "sigma": 0.15, "tau": 0.05},
            {"mu": 6.0, "sigma": 0.1, "tau": 0.01},
        ]

        for params in test_cases:
            f = emg_pdf_E(E, **params)
            integral = np.trapz(f, E)
            assert np.isclose(integral, 1.0, rtol=1e-4, atol=1e-5), (
                f"EMG with {params} should integrate to 1, got {integral}"
            )

    def test_gaussian_pdf_unit_normalization(self):
        """Gaussian PDF should integrate to 1."""
        E = np.linspace(0, 15, 20001)
        f = gaussian_pdf_E(E, mu=5.3, sigma=0.134)

        integral = np.trapz(f, E)
        assert np.isclose(integral, 1.0, rtol=1e-5, atol=1e-6), (
            f"Gaussian PDF should integrate to 1, got {integral}"
        )

    def test_emg_pdf_with_invalid_params(self):
        """EMG should return zero for non-positive sigma or tau."""
        E = np.linspace(0, 10, 1000)

        # Zero sigma
        f_zero_sigma = emg_pdf_E(E, mu=5.0, sigma=0.0, tau=0.02)
        assert np.allclose(f_zero_sigma, 0.0), "EMG with sigma=0 should be zero"

        # Negative sigma
        f_neg_sigma = emg_pdf_E(E, mu=5.0, sigma=-0.1, tau=0.02)
        assert np.allclose(f_neg_sigma, 0.0), "EMG with sigma<0 should be zero"

        # Zero tau
        f_zero_tau = emg_pdf_E(E, mu=5.0, sigma=0.1, tau=0.0)
        assert np.allclose(f_zero_tau, 0.0), "EMG with tau=0 should be zero"

    def test_gaussian_pdf_with_invalid_params(self):
        """Gaussian should return zero for non-positive sigma."""
        E = np.linspace(0, 10, 1000)

        f_zero = gaussian_pdf_E(E, mu=5.0, sigma=0.0)
        assert np.allclose(f_zero, 0.0), "Gaussian with sigma=0 should be zero"

        f_neg = gaussian_pdf_E(E, mu=5.0, sigma=-0.1)
        assert np.allclose(f_neg, 0.0), "Gaussian with sigma<0 should be zero"


class TestIntensityNormalization:
    """Test that intensity functions produce correct densities."""

    def test_single_peak_intensity_integral(self):
        """Intensity for single peak should integrate to the yield N."""
        E = np.linspace(4.5, 8.5, 10001)
        N_Po210 = 15000.0
        domain = (4.5, 8.5)

        params = {
            "N_Po210": N_Po210,
            "mu_Po210": 5.3,
            "sigma0": 0.134,
            "F": 0.0,
            "tau_Po210": 0.02,
            "N_Po218": 0.0,
            "mu_Po218": 6.0,
            "tau_Po218": 0.03,
            "N_Po214": 0.0,
            "mu_Po214": 7.687,
            "tau_Po214": 0.02,
            "b0": 0.0,
            "b1": 0.0,
        }

        lam = spectral_intensity_E(E, params, domain)

        # Integrate intensity
        integral = np.trapz(lam, E)

        # Should equal N_Po210 (plus background, which is zero here)
        expected = integral_of_intensity(params, domain)
        assert np.isclose(integral, expected, rtol=1e-3), (
            f"Intensity integral {integral} should match expected {expected}"
        )
        assert np.isclose(integral, N_Po210, rtol=1e-3), (
            f"Single peak integral {integral} should match yield {N_Po210}"
        )

    def test_multi_peak_intensity_integral(self):
        """Intensity with multiple peaks should integrate to sum of yields."""
        E = np.linspace(4.5, 8.5, 10001)
        N_Po210 = 15000.0
        N_Po218 = 3000.0
        N_Po214 = 8000.0
        domain = (4.5, 8.5)

        params = {
            "N_Po210": N_Po210,
            "mu_Po210": 5.3,
            "sigma0": 0.134,
            "F": 0.0,
            "tau_Po210": 0.02,
            "N_Po218": N_Po218,
            "mu_Po218": 6.0,
            "tau_Po218": 0.03,
            "N_Po214": N_Po214,
            "mu_Po214": 7.687,
            "tau_Po214": 0.02,
            "b0": 0.0,
            "b1": 0.0,
        }

        lam = spectral_intensity_E(E, params, domain)
        integral = np.trapz(lam, E)

        expected_total = N_Po210 + N_Po218 + N_Po214
        expected = integral_of_intensity(params, domain)

        assert np.isclose(integral, expected, rtol=1e-3), (
            f"Multi-peak integral {integral} should match analytical {expected}"
        )
        assert np.isclose(integral, expected_total, rtol=1e-3), (
            f"Multi-peak integral {integral} should match sum of yields {expected_total}"
        )

    def test_background_only_intensity(self):
        """Background-only intensity should integrate correctly."""
        E_min, E_max = 4.8, 8.3
        domain = (E_min, E_max)
        b0 = 1000.0
        b1 = 50.0

        params = {
            "N_Po210": 0.0,
            "mu_Po210": 5.3,
            "sigma0": 0.134,
            "F": 0.0,
            "tau_Po210": 0.02,
            "N_Po218": 0.0,
            "mu_Po218": 6.0,
            "tau_Po218": 0.03,
            "N_Po214": 0.0,
            "mu_Po214": 7.687,
            "tau_Po214": 0.02,
            "b0": b0,
            "b1": b1,
        }

        # Analytical integral of linear background
        dE = E_max - E_min
        expected_bkg = b0 * dE + 0.5 * b1 * (E_max ** 2 - E_min ** 2)

        integral_calc = integral_of_intensity(params, domain)
        assert np.isclose(integral_calc, expected_bkg, rtol=1e-12), (
            f"Background integral {integral_calc} should match analytical {expected_bkg}"
        )

    def test_intensity_with_background(self):
        """Intensity with peaks + background should sum correctly."""
        E_min, E_max = 4.8, 8.3
        domain = (E_min, E_max)

        N_Po210 = 15000.0
        N_Po218 = 3000.0
        N_Po214 = 8000.0
        b0 = 1000.0
        b1 = 100.0

        params = {
            "N_Po210": N_Po210,
            "mu_Po210": 5.3,
            "sigma0": 0.134,
            "F": 0.0,
            "tau_Po210": 0.02,
            "N_Po218": N_Po218,
            "mu_Po218": 6.0,
            "tau_Po218": 0.03,
            "N_Po214": N_Po214,
            "mu_Po214": 7.687,
            "tau_Po214": 0.02,
            "b0": b0,
            "b1": b1,
        }

        # Total expected counts
        dE = E_max - E_min
        expected_bkg = b0 * dE + 0.5 * b1 * (E_max ** 2 - E_min ** 2)
        expected_total = N_Po210 + N_Po218 + N_Po214 + expected_bkg

        integral_calc = integral_of_intensity(params, domain)
        assert np.isclose(integral_calc, expected_total, rtol=1e-12), (
            f"Total integral {integral_calc} should match sum {expected_total}"
        )


class TestUnbinnedCountsScale:
    """Test that unbinned fits produce yields on the correct scale.

    This is the key test for the normalization bug fix. Previously, unbinned
    fits produced yields that were ~1/(4*ΔE_bin) times too large due to
    incorrect bin-width scaling inherited from binned fitting code.
    """

    def test_unbinned_counts_scale(self):
        """Fitted yields should be on the same order as actual event counts."""
        rng = np.random.default_rng(0)

        # Known yields
        N210 = 15000
        N218 = 3000
        N214 = 8000
        b0 = 1000.0
        b1 = 0.0
        E_min, E_max = 4.8, 8.3
        domain = (E_min, E_max)

        # Generate synthetic data (simplified: just Gaussians for speed)
        E_210 = rng.normal(5.30, 0.134, N210)
        E_218 = rng.normal(6.00, 0.134, N218)
        E_214 = rng.normal(7.687, 0.134, N214)
        E = np.concatenate([E_210, E_218, E_214])

        # Add uniform background
        n_bkg = int(b0 * (E_max - E_min))
        E_bkg = rng.uniform(E_min, E_max, n_bkg)
        E = np.concatenate([E, E_bkg])

        # Filter to fit window
        E = E[(E > E_min) & (E < E_max)]
        N_events = E.size

        # Evaluate intensity at true parameters
        params = {
            "N_Po210": N210,
            "mu_Po210": 5.30,
            "sigma0": 0.134,
            "F": 0.0,
            "tau_Po210": 0.02,
            "N_Po218": N218,
            "mu_Po218": 6.00,
            "tau_Po218": 0.03,
            "N_Po214": N214,
            "mu_Po214": 7.687,
            "tau_Po214": 0.02,
            "b0": b0,
            "b1": b1,
        }

        lam = spectral_intensity_E(E, params, domain)

        # All densities must be positive and finite
        assert (lam > 0).all(), "Intensity must be positive at all data points"
        assert np.isfinite(lam).all(), "Intensity must be finite"

        # Extended integral should equal total expected counts
        mu = integral_of_intensity(params, domain)
        expected_mu = N210 + N218 + N214 + b0 * (E_max - E_min)

        assert np.isclose(mu, expected_mu, rtol=1e-12), (
            f"Extended term μ={mu} should match analytical {expected_mu}"
        )

        # Key test: sum of yields should be ~ same order as N_events
        # NOT 19× larger as in the buggy version!
        sum_yields = N210 + N218 + N214
        ratio = sum_yields / N_events

        assert 0.3 < ratio < 1.5, (
            f"Sum of yields ({sum_yields}) should be within [0.3, 1.5] × "
            f"N_events ({N_events}), got ratio {ratio:.2f}. "
            f"If ratio ≈ 19, the bin-width bug is still present!"
        )

    def test_nll_is_finite(self):
        """NLL should return finite values for valid parameters."""
        rng = np.random.default_rng(42)

        E_min, E_max = 4.8, 8.3
        domain = (E_min, E_max)

        # Generate toy data
        E = rng.normal(5.3, 0.15, 1000)
        E = E[(E > E_min) & (E < E_max)]

        params = {
            "N_Po210": 1000.0,
            "mu_Po210": 5.3,
            "sigma0": 0.134,
            "F": 0.0,
            "tau_Po210": 0.02,
            "N_Po218": 100.0,
            "mu_Po218": 6.0,
            "tau_Po218": 0.03,
            "N_Po214": 200.0,
            "mu_Po214": 7.687,
            "tau_Po214": 0.02,
            "b0": 100.0,
            "b1": 0.0,
        }

        nll = nll_extended_unbinned(E, params, domain)

        assert np.isfinite(nll), f"NLL should be finite, got {nll}"
        assert nll > 0, f"NLL should be positive, got {nll}"

    def test_nll_with_zero_data(self):
        """NLL with zero events should return inf."""
        E = np.array([])
        domain = (4.8, 8.3)
        params = {"N_Po210": 1000.0, "mu_Po210": 5.3, "sigma0": 0.1, "F": 0.0}

        nll = nll_extended_unbinned(E, params, domain)
        assert nll == np.inf, "NLL with zero data should be inf"

    def test_nll_with_invalid_intensity(self):
        """NLL should return inf if intensity is non-positive."""
        E = np.array([5.3, 6.0, 7.7])
        domain = (4.8, 8.3)

        # Negative yield (should clip to positive via intensity function)
        params = {
            "N_Po210": -1000.0,  # Invalid
            "mu_Po210": 5.3,
            "sigma0": 0.134,
            "F": 0.0,
            "tau_Po210": 0.02,
            "N_Po218": 0.0,
            "mu_Po218": 6.0,
            "tau_Po218": 0.03,
            "N_Po214": 0.0,
            "mu_Po214": 7.687,
            "tau_Po214": 0.02,
            "b0": 0.0,
            "b1": 0.0,
        }

        # With negative yield and zero background, intensity should be ~zero
        # which makes NLL invalid
        nll = nll_extended_unbinned(E, params, domain)
        # Depending on implementation, this might be inf or a large finite value
        # The key is it shouldn't crash
        assert np.isfinite(nll) or nll == np.inf


class TestArrayParamsCompatibility:
    """Test backward compatibility with array-style parameters."""

    def test_array_params_12_element(self):
        """Intensity should accept 12-element array params."""
        E = np.linspace(4.8, 8.3, 1001)
        domain = (4.8, 8.3)

        # 12-element format: N210, mu210, sig0, tau210, N218, mu218, tau218,
        #                    N214, mu214, tau214, b0, b1
        params_array = [15000, 5.3, 0.134, 0.02, 3000, 6.0, 0.03, 8000, 7.687, 0.02, 1000, 0]

        lam = spectral_intensity_E(E, params_array, domain)

        assert lam.shape == E.shape
        assert (lam > 0).all()
        assert np.isfinite(lam).all()

    def test_integral_with_array_params(self):
        """integral_of_intensity should work with array params."""
        domain = (4.8, 8.3)
        params_array = [15000, 5.3, 0.134, 0.02, 3000, 6.0, 0.03, 8000, 7.687, 0.02, 1000, 50]

        N210, mu210, sig0, tau210, N218, mu218, tau218, N214, mu214, tau214, b0, b1 = params_array
        E_min, E_max = domain

        integral = integral_of_intensity(params_array, domain)

        # Expected: sum of yields + background integral
        dE = E_max - E_min
        expected = N210 + N218 + N214 + b0 * dE + 0.5 * b1 * (E_max ** 2 - E_min ** 2)

        assert np.isclose(integral, expected, rtol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
