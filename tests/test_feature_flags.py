import numpy as np
import pytest
from types import SimpleNamespace

from fitting import fit_spectrum
import likelihood_ext


def _short_spectrum():
    rng = np.random.default_rng(0)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 20),
        rng.normal(6.0, 0.05, 20),
        rng.normal(7.7, 0.05, 20),
    ])
    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (20, 5),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (20, 5),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (20, 5),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }
    return energies, priors


def test_background_linear_noop():
    energies, priors = _short_spectrum()
    out_default = fit_spectrum(energies, priors)
    out_linear = fit_spectrum(energies, priors, opts=SimpleNamespace(background_model="linear"))
    for key in ("b0", "b1", "S_bkg"):
        assert out_default.params[key] == pytest.approx(out_linear.params[key])


def test_likelihood_extended_switch(monkeypatch):
    energies, priors = _short_spectrum()
    called = {}
    orig = likelihood_ext.neg_loglike_extended

    def wrapped(E, intensity_fn, params, *, area_keys, clip=1e-300):
        called["called"] = True
        return orig(E, intensity_fn, params, area_keys=area_keys, clip=clip)

    monkeypatch.setattr(likelihood_ext, "neg_loglike_extended", wrapped)

    fit_spectrum(energies, priors, unbinned=True)
    assert "called" not in called

    called.clear()
    fit_spectrum(
        energies,
        priors,
        unbinned=True,
        opts=SimpleNamespace(likelihood="extended"),
    )
    assert called.get("called")
