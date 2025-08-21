"""Utility math functions."""

from __future__ import annotations

import numpy as np


def log_expm1_stable(y: np.ndarray | float) -> np.ndarray | float:
    """Return ``log(expm1(y))`` computed in a numerically stable way.

    Parameters
    ----------
    y : array-like
        Input values.

    Returns
    -------
    array-like
        ``log(expm1(y))`` evaluated element-wise with stability fixes.

    Notes
    -----
    For ``y > 0`` the transformation ``y + log1p(-exp(-y))`` is used to avoid
    overflow of ``expm1``. For ``y <= 0`` we evaluate ``log(expm1(y))`` but
    clamp the argument to ``>= tiny`` to avoid ``log(0)``. The function never
    returns ``NaN`` for finite ``y`` and is monotonic in ``y``.
    """

    y = np.asarray(y, dtype=float)
    out = np.empty_like(y, dtype=float)
    pos = y > 0
    # y + log1p(-exp(-y)) is stable for large positive y
    out[pos] = y[pos] + np.log1p(-np.exp(-y[pos]))
    # For y <= 0 use direct evaluation with a clamp to avoid log(0)
    neg = ~pos
    if np.any(neg):
        val = np.expm1(y[neg])
        tiny = np.finfo(float).tiny
        val = np.maximum(val, tiny)
        out[neg] = np.log(val)
    return out
