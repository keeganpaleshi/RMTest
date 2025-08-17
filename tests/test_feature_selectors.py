import numpy as np
import pytest

from fitting import fit_spectrum


def _generate_energies(seed: int = 0):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(5.3, 0.05, 200),
        rng.normal(6.0, 0.05, 200),
        rng.normal(7.7, 0.05, 200),
    ])


def _base_priors():
    return {
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


def test_default_vs_explicit_linear_identical():
    energies = _generate_energies(0)
    priors = _base_priors()
    res_default = fit_spectrum(energies, priors)
    res_linear = fit_spectrum(energies, priors, flags={"background_model": "linear"})
    for key, val in res_default.params.items():
        assert res_linear.params[key] == pytest.approx(val)


def test_explicit_extended_matches_default():
    energies = _generate_energies(1)
    priors = _base_priors()
    flags = {
        "fix_S_Po210": True,
        "fix_S_Po218": True,
        "fix_S_Po214": True,
        "fix_b0": True,
        "fix_b1": True,
    }
    res_default = fit_spectrum(energies, priors, unbinned=True, flags=flags)
    flags_ext = dict(flags)
    flags_ext["likelihood"] = "extended"
    res_ext = fit_spectrum(energies, priors, unbinned=True, flags=flags_ext)
    for key in ("mu_Po210", "mu_Po218", "mu_Po214"):
        assert res_ext.params[key] == pytest.approx(res_default.params[key], rel=5e-2)
