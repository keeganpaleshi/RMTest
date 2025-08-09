"""Extended unbinned negative log-likelihood utilities.

This module defines an opt-in API for computing the extended
negative log-likelihood (NLL) for unbinned energy spectra.  The
implementation is intentionally lightweight and depends only on
:mod:`numpy`.

Example
-------
The user supplies an intensity function ``λ(E)`` (in counts/MeV) and a
parameter mapping.  The function below represents a flat unit-rate
background combined with an area parameter for a single spectral
component::

    >>> import numpy as np
    >>> def flat_background(E, params):
    ...     return np.ones_like(E) * params["rate"]
    >>> params = {"rate": 1.0, "A": 0.0}
    >>> neg_loglike_extended(np.array([1.0, 2.0, 3.0]), flat_background, params,
    ...                      area_keys=["A"])

The function returns the scalar NLL and performs no fitting or
minimisation on its own.
"""

from __future__ import annotations

import numpy as np

__all__ = ["neg_loglike_extended"]


def _softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable softplus transformation."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def neg_loglike_extended(E, intensity_fn, params, *, area_keys, clip=1e-300):
    """Extended unbinned negative log-likelihood.

    Parameters
    ----------
    E : array_like
        Unbinned energies in MeV.
    intensity_fn : callable
        Callable ``intensity_fn(E, params)`` returning ``λ(E)`` in counts/MeV.
        It may, for example, incorporate backgrounds generated with
        :func:`fitting._make_linear_bkg` or similar utilities.
    params : Mapping
        Parameter dictionary or similar object providing values for
        ``intensity_fn`` and ``area_keys``.
    area_keys : sequence
        Parameter names whose (softplus-mapped) values contribute to the
        expected event count ``Nexp``.
    clip : float, optional
        Lower bound applied to ``λ(E)`` to maintain numerical stability.

    Returns
    -------
    float
        The scalar negative log-likelihood value.
    """
    E = np.asarray(E, dtype=float)
    lam = np.clip(intensity_fn(E, params), clip, np.inf)
    Nexp = sum(_softplus(params[k]) for k in area_keys)
    return float(-(np.sum(np.log(lam)) - Nexp))
