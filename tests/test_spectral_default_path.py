
import warnings
import numpy as np
import yaml
from scipy.optimize import OptimizeWarning
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
