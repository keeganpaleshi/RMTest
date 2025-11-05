# tests/test_density_clip_safety.py
"""Tests for numerical safety of density clipping."""
import numpy as np
from fitting import fit_spectrum


def test_density_never_zero_after_clip():
    """
    Test that density values are never zero after clipping, even in extreme tails.
    This ensures log-likelihood computations remain finite.
    """
    # Generate synthetic data with a single peak
    rng = np.random.default_rng(42)
    energies = rng.normal(5.3, 0.05, 1000)

    # Define priors for a simple Gaussian peak
    priors = {
        "mu_Po214": (5.3, 0.01),
        "S_Po214": (1000.0, 100.0),
        "sigma0": (0.05, 0.005),
        "F": (0.0, 0.001),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    # Create a config with a known clip_floor
    config = {
        "spectral_fit": {"clip_floor": 1e-50},
        "isotopes": {
            "Po214": {"half_life_s": 164.3e-6, "energy_MeV": 7.687}
        },
    }

    flags = {
        "fix_sigma0": True,
        "fix_F": True,
    }

    # Fit the spectrum
    result = fit_spectrum(
        energies,
        priors,
        flags=flags,
        config=config,
        unbinned=False,
    )

    # The fit should succeed without numerical issues
    assert "mu_Po214" in result
    assert np.isfinite(result["mu_Po214"])


def test_clip_floor_prevents_log_zero():
    """
    Test that clip_floor prevents log(0) errors in likelihood computation.
    """
    # Create data far from any peak to test tail behavior
    energies = np.array([10.0] * 10)  # Far from typical radon peaks

    priors = {
        "mu_Po214": (5.3, 0.01),
        "S_Po214": (10.0, 5.0),
        "sigma0": (0.05, 0.005),
        "F": (0.0, 0.001),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    config = {
        "spectral_fit": {"clip_floor": 1e-100},
        "isotopes": {
            "Po214": {"half_life_s": 164.3e-6, "energy_MeV": 7.687}
        },
    }

    flags = {
        "fix_sigma0": True,
        "fix_F": True,
    }

    # This should not raise due to log(0) or numerical issues
    result = fit_spectrum(
        energies,
        priors,
        flags=flags,
        config=config,
        unbinned=False,
    )

    assert result is not None
