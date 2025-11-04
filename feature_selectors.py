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


def _existing_linear_bkg(E, params):
    """Legacy linear background model returning counts/MeV."""

    b0 = params.get("b0", 0.0)
    b1 = params.get("b1", 0.0)
    return b0 + b1 * np.asarray(E)


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
        required = {"S_bkg", "b0", "b1"}

        def bkg(E, params):
            missing = required - params.keys()
            if missing:
                got = sorted(params.keys())
                raise ValueError(
                    "background_model=loglin_unit requires params {S_bkg, b0, b1}; got: "
                    f"{got}"
                )
            val = shape(E, params["b0"], params["b1"])
            return softplus(params["S_bkg"]) * val

        return bkg

    return lambda E, params: _existing_linear_bkg(E, params)


def select_neg_loglike(opts: Any) -> Callable:
    """Return the negative log-likelihood function based on ``opts``."""

    if getattr(opts, "likelihood", "") == "extended":
        from likelihood_ext import neg_loglike_extended as _neg_loglike_extended

        def neg_loglike(
            E: Sequence[float],
            intensity_fn: Callable[[Sequence[float], Mapping[str, float]], np.ndarray],
            params: Mapping[str, float],
            *,
            area_keys: Sequence[str],
            clip: float = 1e-300,
            background_model: str | None = None,
            bounds: tuple[float, float] | None = None,
            expected_total: float | None = None,
        ) -> float:
            missing = [k for k in area_keys if k not in params]
            if missing:
                got = sorted(params.keys())
                needed = ", ".join(area_keys)
                raise ValueError(
                    f"likelihood=extended requires params {{{needed}}}; got: {got}"
                )
            return _neg_loglike_extended(
                E,
                intensity_fn,
                params,
                area_keys=area_keys,
                clip=clip,
                background_model=background_model,
                bounds=bounds,
                expected_total=expected_total,
            )

        return neg_loglike

    def neg_loglike_current(
        E: Sequence[float],
        intensity_fn: Callable[[Sequence[float], Mapping[str, float]], np.ndarray],
        params: Mapping[str, float],
        *,
        area_keys: Sequence[str] | None = None,
        clip: float = 1e-300,
        bounds: tuple[float, float] | None = None,
        expected_total: float | None = None,
    ) -> float:
        """Simple unextended unbinned negative log-likelihood."""

        lam = np.clip(intensity_fn(E, params), clip, np.inf)
        return float(-np.sum(np.log(lam)))

    return neg_loglike_current


__all__ = ["select_background_factory", "select_neg_loglike"]

