import numpy as np


def log_expm1_stable(y: np.ndarray) -> np.ndarray:
    """Return ``log(expm1(y))`` with improved stability.

    Parameters
    ----------
    y : np.ndarray
        Input array of finite values.

    Returns
    -------
    np.ndarray
        Element-wise ``log(expm1(y))`` computed in a numerically stable way.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask = y > 0
    # For positive values use y + log1p(-exp(-y)) to avoid overflow
    out[mask] = y[mask] + np.log1p(-np.exp(-y[mask]))
    # For non-positive values use log(expm1(y)) with argument clamped away from zero
    tiny = np.finfo(y.dtype).tiny
    expm1_vals = np.expm1(y[~mask])
    expm1_vals = np.clip(expm1_vals, tiny, None)
    out[~mask] = np.log(expm1_vals)
    return out
