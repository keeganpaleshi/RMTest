"""Utilities for extended unbinned likelihood fits.

This module provides an opt-in implementation of an extended
unbinned negative log-likelihood.  It is not used by default in the
existing fitting pipeline but serves as a reference for future work.

Examples
--------
>>> import numpy as np
>>> from likelihood_ext import neg_loglike_extended
>>> def flat_intensity(E, params):
...     # Reference intensity returning a flat spectrum
...     return np.full_like(E, params["rate"])
>>> params = {"rate": 0.2, "S": -1.0}
>>> E = np.array([1.0, 2.0])
>>> round(neg_loglike_extended(E, flat_intensity, params, area_keys=["S"]), 5)
0.59861
"""

from __future__ import annotations

import numpy as np

__all__ = ["neg_loglike_extended"]


def _softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable softplus transformation."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def neg_loglike_extended(
    E: np.ndarray,
    intensity_fn,
    params,
    *,
    area_keys,
    clip: float = 1e-300,
) -> float:
    """Extended unbinned negative log-likelihood.

    Parameters
    ----------
    E : array-like
        Unbinned energies (MeV).
    intensity_fn : callable
        ``intensity_fn(E, params) -> λ(E)`` in counts/MeV.
    params : mapping
        Parameters supplied to ``intensity_fn``.
    area_keys : iterable of str
        Keys in ``params`` whose softplus corresponds to expected areas.
    clip : float, optional
        Lower bound for the intensity to avoid ``log(0)`` (default: ``1e-300``).

    Returns
    -------
    float
        Negative log-likelihood value.
    """

    lam = np.clip(intensity_fn(E, params), clip, np.inf)
    Nexp = sum(_softplus(params[k]) for k in area_keys)
    return float(-(np.sum(np.log(lam)) - Nexp))
