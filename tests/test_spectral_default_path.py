import copy
import warnings

import numpy as np
import yaml
from scipy.optimize import OptimizeWarning

import analyze


def _run_fit(cfg):
    energies = np.array([5.3, 6.0, 7.7])
    priors = {
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (1.0, 0.1),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (1.0, 0.1),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (1.0, 0.1),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        res, _ = analyze._spectral_fit_with_check(
            energies,
            priors,
            {},
            cfg,
            unbinned=cfg["spectral_fit"].get("unbinned_likelihood", False),
        )
    return res.params["likelihood_path"]


def test_spectral_likelihood_path_default_and_unbinned():
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    # Default config uses binned Poisson path
    default_path = _run_fit(cfg)
    assert default_path == "binned_poisson"

    # Enabling unbinned_likelihood flips to unbinned path
    cfg_unbinned = copy.deepcopy(cfg)
    cfg_unbinned["spectral_fit"]["unbinned_likelihood"] = True
    unbinned_path = _run_fit(cfg_unbinned)
    assert unbinned_path in {"unbinned", "unbinned_extended"}
