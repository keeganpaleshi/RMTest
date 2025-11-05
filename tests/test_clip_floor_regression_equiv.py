# tests/test_clip_floor_regression_equiv.py
"""Regression equivalence tests for clip_floor parameter."""
import numpy as np
from fitting import fit_spectrum


def test_clip_floor_regression_equivalence():
    """
    Test that different clip_floor values produce equivalent results
    inside the fitting window where density is significant.
    """
    rng = np.random.default_rng(7)

    # Create synthetic multi-peak spectrum
    energies = np.concatenate([
        rng.normal(5.30, 0.05, 300),
        rng.normal(6.00, 0.05, 400),
        rng.normal(7.70, 0.05, 300),
    ])

    priors = {
        "mu_Bi214": (5.30, 0.01),
        "S_Bi214": (300.0, 50.0),
        "mu_Po218": (6.00, 0.01),
        "S_Po218": (400.0, 50.0),
        "mu_Po214": (7.70, 0.01),
        "S_Po214": (300.0, 50.0),
        "sigma0": (0.05, 0.005),
        "F": (0.0, 0.001),
        "b0": (0.2, 0.1),
        "b1": (0.0, 0.1),
    }

    flags = {
        "fix_sigma0": True,
        "fix_F": True,
    }

    isotopes = {
        "Bi214": {"half_life_s": 1194.0, "energy_MeV": 5.617},
        "Po218": {"half_life_s": 186.0, "energy_MeV": 6.115},
        "Po214": {"half_life_s": 164.3e-6, "energy_MeV": 7.687},
    }

    # Fit with default clip_floor (1e-300)
    config1 = {
        "spectral_fit": {"clip_floor": 1e-300},
        "isotopes": isotopes,
    }
    result1 = fit_spectrum(
        energies,
        priors,
        flags=flags,
        config=config1,
        unbinned=False,
    )

    # Fit with a different clip_floor (1e-100)
    config2 = {
        "spectral_fit": {"clip_floor": 1e-100},
        "isotopes": isotopes,
    }
    result2 = fit_spectrum(
        energies,
        priors,
        flags=flags,
        config=config2,
        unbinned=False,
    )

    # The fitted parameters should be very close
    # Clip floor should not meaningfully change the model inside the window
    for key in ["mu_Bi214", "mu_Po218", "mu_Po214", "S_Bi214", "S_Po218", "S_Po214"]:
        if key in result1 and key in result2:
            val1 = result1[key]
            val2 = result2[key]
            # Allow for numerical differences but they should be very small
            rel_diff = abs(val1 - val2) / max(abs(val1), 1e-10)
            assert rel_diff < 1e-3, f"{key}: {val1} vs {val2}, rel_diff={rel_diff}"


def test_clip_floor_no_config_uses_default():
    """
    Test that when no config is provided, the default clip_floor (1e-300) is used.
    """
    rng = np.random.default_rng(42)
    energies = rng.normal(5.3, 0.05, 500)

    priors = {
        "mu_Po214": (5.3, 0.01),
        "S_Po214": (500.0, 50.0),
        "sigma0": (0.05, 0.005),
        "F": (0.0, 0.001),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    flags = {
        "fix_sigma0": True,
        "fix_F": True,
    }

    # Fit without config (should use default 1e-300)
    result_no_config = fit_spectrum(
        energies,
        priors,
        flags=flags,
        config=None,
        unbinned=False,
    )

    # Fit with explicit default
    config_explicit = {
        "spectral_fit": {"clip_floor": 1e-300},
        "isotopes": {
            "Po214": {"half_life_s": 164.3e-6, "energy_MeV": 7.687}
        },
    }
    result_explicit = fit_spectrum(
        energies,
        priors,
        flags=flags,
        config=config_explicit,
        unbinned=False,
    )

    # Results should be identical
    assert abs(result_no_config["mu_Po214"] - result_explicit["mu_Po214"]) < 1e-9
