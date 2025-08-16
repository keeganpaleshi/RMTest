"""Helper selectors for experimental features."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


__all__ = ["select_background_factory", "select_neg_loglike"]


def existing_linear_bkg(E: np.ndarray | float, params: Mapping[str, float]) -> np.ndarray | float:
    """Simple linear background ``b0 + b1 * E``."""
    b0 = float(params.get("b0", 0.0))
    b1 = float(params.get("b1", 0.0))
    E = np.asarray(E, dtype=float)
    return b0 + b1 * E


def select_background_factory(opts: Any, Emin: float, Emax: float) -> Callable[[np.ndarray, Mapping[str, float]], np.ndarray]:
    """Return a background intensity function based on ``opts.background_model``.

    Parameters
    ----------
    opts:
        Object with attribute ``background_model``.
    Emin, Emax:
        Energy bounds for the log-linear shape.
    """
    model = getattr(opts, "background_model", "linear")
    if model == "loglin_unit":
        from fitting import make_linear_bkg, softplus

        shape_fn = make_linear_bkg(Emin, Emax)

        def bkg(E, params):
            beta0 = params.get("beta0", params.get("b0", 0.0))
            beta1 = params.get("beta1", params.get("b1", 0.0))
            shape = shape_fn(E, beta0, beta1)
            return softplus(params["S_bkg"]) * shape

        return bkg
    else:
        return lambda E, params: existing_linear_bkg(E, params)


def existing_neg_loglike(
    E: np.ndarray,
    intensity_fn: Callable[[np.ndarray, Mapping[str, float]], np.ndarray],
    params: Mapping[str, float],
    *,
    area_keys,
    clip: float = 1e-300,
) -> float:
    """Status-quo negative log-likelihood ignoring expected counts."""
    lam = np.clip(intensity_fn(E, params), clip, np.inf)
    return float(-np.sum(np.log(lam)))


def select_neg_loglike(opts: Any) -> Callable:
    """Select the negative log-likelihood implementation."""
    if getattr(opts, "likelihood", "current") == "extended":
        from likelihood_ext import neg_loglike_extended

        return neg_loglike_extended
    return existing_neg_loglike
