
import warnings

import numpy as np
import yaml
from scipy.optimize import OptimizeWarning

import fitting
from fitting import fit_spectrum


def _minimal_priors():
    return {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (10, 1),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (10, 1),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (10, 1),
        "b0": (0.0, 0.1),
        "b1": (0.0, 0.1),
    }


def _run_fit(unbinned: bool) -> str:
    priors = _minimal_priors()
    energies = np.concatenate([np.full(5, 5.3), np.full(5, 6.0), np.full(5, 7.7)])
    flags = {"fix_sigma0": True, "fix_F": True}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        result = fit_spectrum(energies, priors, flags=flags, unbinned=unbinned)
    return result.params.get("likelihood_path")


def test_default_and_unbinned_likelihood_path():
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    path = _run_fit(cfg["spectral_fit"].get("unbinned_likelihood", False))
    assert path == "binned_poisson"

    cfg2 = yaml.safe_load(open("config.yaml"))
    cfg2.setdefault("spectral_fit", {})["unbinned_likelihood"] = True
    path_unbinned = _run_fit(cfg2["spectral_fit"].get("unbinned_likelihood", False))
    assert path_unbinned in {"unbinned", "unbinned_extended"}


def test_binned_fit_skips_original_curve_fit(monkeypatch):
    priors = _minimal_priors()
    energies = np.concatenate([np.full(5, 5.3), np.full(5, 6.0), np.full(5, 7.7)])
    flags = {"fix_sigma0": True, "fix_F": True}

    def _fail_curve_fit(*args, **kwargs):
        raise AssertionError("original curve_fit should not be called")

    monkeypatch.setattr(fitting, "curve_fit", _fail_curve_fit)
    monkeypatch.setattr(fitting, "_ORIG_CURVE_FIT", _fail_curve_fit)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        fit_spectrum(energies, priors, flags=flags, unbinned=False)


def test_alternate_curve_fit_values_used(monkeypatch):
    priors = _minimal_priors()
    energies = np.concatenate([np.full(5, 5.3), np.full(5, 6.0), np.full(5, 7.7)])
    flags = {"fix_sigma0": True, "fix_F": True}
    sentinel_mu = 5.1234
    captured = {}

    def _fake_curve_fit(model, xdata, ydata, p0=None, bounds=None):
        captured["p0"] = list(p0) if p0 is not None else []
        captured["bounds"] = bounds
        popt = np.array(p0, dtype=float)
        popt[0] = sentinel_mu
        pcov = np.eye(len(popt))
        return popt, pcov

    monkeypatch.setattr(fitting, "curve_fit", _fake_curve_fit)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        result = fit_spectrum(energies, priors, flags=flags, unbinned=False)

    assert np.isclose(result.params["mu_Po210"], sentinel_mu)
    assert np.isclose(result.cov[result.param_index["mu_Po210"], result.param_index["mu_Po210"]], 1.0)
    assert len(captured["p0"]) == len(captured["bounds"][0])
