import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Ensure repository root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_selectors import select_background_factory, select_neg_loglike
from fitting import fit_spectrum
from tests.synthetic_dataset import synthetic_spectrum


def _priors():
    return {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (50, 10),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (60, 10),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (70, 10),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }


def test_background_factory_linear_default_identical():
    energies = synthetic_spectrum(rng_seed=0)
    priors = _priors()
    res_default = fit_spectrum(energies, dict(priors))

    opts = SimpleNamespace()
    bkg = select_background_factory(opts, float(energies.min()), float(energies.max()))
    res_explicit = fit_spectrum(energies, dict(priors), background_factory=bkg)

    for key in (
        "b0",
        "b1",
        "mu_Po210",
        "S_Po210",
        "mu_Po218",
        "S_Po218",
        "mu_Po214",
        "S_Po214",
    ):
        assert np.isclose(res_default.params[key], res_explicit.params[key])


def test_likelihood_extended_smoke():
    energies = synthetic_spectrum(rng_seed=2)
    priors = _priors()
    priors["S_bkg"] = (100, 10)
    flags = {"background_model": "loglin_unit"}
    res_default = fit_spectrum(energies, dict(priors), unbinned=True, flags=flags)

    opts = SimpleNamespace(likelihood="extended", background_model="loglin_unit")
    nll = select_neg_loglike(opts)
    bkg = select_background_factory(opts, float(energies.min()), float(energies.max()))
    res_ext = fit_spectrum(
        energies,
        dict(priors),
        unbinned=True,
        flags=flags,
        background_factory=bkg,
        neg_loglike=nll,
    )

    for key in (
        "b0",
        "b1",
        "mu_Po210",
        "S_Po210",
        "mu_Po218",
        "S_Po218",
        "mu_Po214",
        "S_Po214",
        "S_bkg",
    ):
        assert np.isclose(res_default.params[key], res_ext.params[key], rtol=1e-3, atol=1e-6)
