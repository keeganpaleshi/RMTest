"""Factory helpers for optional background and likelihood models.

These helpers expose experimental feature-flagged behaviour without changing
the default execution path.  Callers can supply an object ``opts`` providing
``background_model`` and ``likelihood`` attributes to opt in to the new
implementations.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fitting import softplus


def select_background_factory(opts: Any, Emin: float, Emax: float) -> Callable:
    """Select a background factory based on ``opts.background_model``.

    Parameters
    ----------
    opts : Any
        Object providing a ``background_model`` attribute.
    Emin, Emax : float
        Energy bounds used when constructing the log-linear shape.
    """

    if getattr(opts, "background_model", "") == "loglin_unit":
        from fitting import make_linear_bkg

        shape = make_linear_bkg(Emin, Emax)

        def bkg(E, params):
            val = shape(E, params.get("beta0", params.get("b0", 0.0)), params.get("beta1", params.get("b1", 0.0)))
            return softplus(params.get("S_bkg", 0.0)) * val

        return bkg

    def legacy_bkg(E, params):
        b0 = params.get("b0", 0.0)
        b1 = params.get("b1", 0.0)
        B = softplus(params.get("S_bkg", 0.0))
        norm = b0 * (Emax - Emin) + 0.5 * b1 * (Emax**2 - Emin**2)
        norm = norm if norm > 0 else 1.0
        return B * (b0 + b1 * np.asarray(E)) / norm

    return legacy_bkg


def select_neg_loglike(opts: Any) -> Callable:
    """Return the negative log-likelihood function based on ``opts``."""

    if getattr(opts, "likelihood", "") == "extended":
        from likelihood_ext import neg_loglike_extended

        return neg_loglike_extended

    def neg_loglike_current(
        E: Sequence[float],
        intensity_fn: Callable[[Sequence[float], Mapping[str, float]], np.ndarray],
        params: Mapping[str, float],
        *,
        area_keys: Sequence[str] | None = None,
        clip: float = 1e-300,
    ) -> float:
        """Simple unextended unbinned negative log-likelihood."""

        lam = np.clip(intensity_fn(E, params), clip, np.inf)
        return float(-np.sum(np.log(lam)))

    return neg_loglike_current


__all__ = ["select_background_factory", "select_neg_loglike"]

