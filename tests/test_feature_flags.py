import numpy as np
import pytest
from types import SimpleNamespace

from fitting import fit_spectrum
import likelihood_ext


def _generate_energies():
    rng = np.random.default_rng(0)
    return np.concatenate([
        rng.normal(5.3, 0.05, 50),
        rng.normal(6.0, 0.05, 50),
        rng.normal(7.7, 0.05, 50),
    ])


def _base_priors():
    return {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (50, 5),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (50, 5),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (50, 5),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }


def test_background_model_linear_matches_default():
    energies = _generate_energies()
    priors = _base_priors()
    res_def = fit_spectrum(energies, priors)
    res_lin = fit_spectrum(energies, priors, opts=SimpleNamespace(background_model="linear"))
    for key in ("b0", "b1"):
        assert res_def.params[key] == pytest.approx(res_lin.params[key])


def test_likelihood_extended_switches(monkeypatch):
    called = {"n": 0}

    def fake_nll(E, intensity_fn, params, *, area_keys, clip=1e-300):
        called["n"] += 1
        return 0.0

    monkeypatch.setattr(likelihood_ext, "neg_loglike_extended", fake_nll)

    energies = _generate_energies()
    priors = {
        "sigma0": (0.05, 0.0),
        "F": (0.0, 0.0),
        "mu_Po210": (5.3, 0.0),
        "S_Po210": (50, 0.0),
        "mu_Po218": (6.0, 0.0),
        "S_Po218": (50, 0.0),
        "mu_Po214": (7.7, 0.0),
        "S_Po214": (50, 0.0),
        "b0": (0.0, 0.0),
        "b1": (0.0, 0.0),
    }
    flags = {f"fix_{k}": True for k in priors}

    fit_spectrum(
        energies,
        priors,
        flags=flags,
        unbinned=True,
        opts=SimpleNamespace(likelihood="extended"),
    )
    assert called["n"] > 0

    called["n"] = 0
    fit_spectrum(energies, priors, flags=flags, unbinned=True)
    assert called["n"] == 0
