import numpy as np
import pytest

import likelihood_ext
from fitting import fit_spectrum
from feature_selectors import select_background_factory, select_neg_loglike
from types import SimpleNamespace


def _generate_energies(seed: int = 0, n: int = 20):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(5.3, 0.05, n),
        rng.normal(6.0, 0.05, n),
        rng.normal(7.7, 0.05, n),
    ])


def _base_priors(n: int = 20):
    return {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (n, 5),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (n, 5),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (n, 5),
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


def test_explicit_extended_matches_default(monkeypatch):
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

    called = {"count": 0}
    orig = likelihood_ext.neg_loglike_extended

    def spy(*args, **kwargs):
        called["count"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(likelihood_ext, "neg_loglike_extended", spy)

    flags_ext = dict(flags)
    flags_ext["likelihood"] = "extended"
    res_ext = fit_spectrum(energies, priors, unbinned=True, flags=flags_ext)

    assert called["count"] > 0
    for key in ("mu_Po210", "mu_Po218", "mu_Po214"):
        assert res_ext.params[key] == pytest.approx(
            res_default.params[key], rel=5e-2
        )


def test_loglin_unit_missing_param():
    opts = SimpleNamespace(background_model="loglin_unit")
    bkg = select_background_factory(opts, 4.0, 8.0)
    with pytest.raises(ValueError) as exc:
        bkg(np.array([5.0]), {"b0": 0.0, "S_bkg": 1.0})
    assert (
        str(exc.value)
        == "background_model=loglin_unit requires params {S_bkg, b0, b1}; missing: ['b1']"
    )


def test_extended_likelihood_missing_area_key():
    opts = SimpleNamespace(likelihood="extended")
    neg_loglike = select_neg_loglike(opts)
    E = np.array([1.0, 2.0, 3.0])
    def intensity(E_vals, params):
        return np.ones_like(E_vals)
    params = {"area": 0.0}
    with pytest.raises(ValueError) as exc:
        neg_loglike(E, intensity, params, area_keys=("area", "missing"))
    assert (
        str(exc.value)
        == "likelihood=extended requires params {area, missing}; missing: ['missing']"
    )
