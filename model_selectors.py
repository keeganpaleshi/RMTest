from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from fitting import softplus


def select_background_factory(opts: Mapping[str, Any], Emin: float, Emax: float) -> Callable[[np.ndarray, Mapping[str, float]], np.ndarray]:
    """Return a background function based on configuration options."""
    model = getattr(opts, "background_model", "linear")
    if model == "loglin_unit":
        from fitting import make_linear_bkg  # lazy import

        def bkg(E, params):
            shape = make_linear_bkg(Emin, Emax)(E, params["beta0"], params["beta1"])
            return softplus(params["S_bkg"]) * shape

        return bkg

    def existing_linear_bkg(E, params):
        beta0 = params["beta0"]
        beta1 = params["beta1"]
        S = params.get("S_bkg", 0.0)
        norm = beta0 * (Emax - Emin) + 0.5 * beta1 * (Emax**2 - Emin**2)
        norm = max(norm, 1e-300)
        return softplus(S) * (beta0 + beta1 * E) / norm

    return existing_linear_bkg


def select_neg_loglike(opts: Mapping[str, Any]):
    """Return the negative log-likelihood function for spectral fits."""
    if getattr(opts, "likelihood", "current") == "extended":
        from likelihood_ext import neg_loglike_extended

        return neg_loglike_extended

    def existing_neg_loglike(E, intensity_fn, params, *, area_keys, clip=1e-300):
        lam = np.clip(intensity_fn(E, params), clip, np.inf)
        Nexp = float(sum(params[k] for k in area_keys))
        return float(-(np.sum(np.log(lam)) - Nexp))

    return existing_neg_loglike
