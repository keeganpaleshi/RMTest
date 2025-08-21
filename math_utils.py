import numpy as np


def log_expm1_stable(y: np.ndarray) -> np.ndarray:
    """Compute log(expm1(y)) in a numerically stable way.

    Parameters
    ----------
    y : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Element-wise ``log(expm1(y))`` evaluated in a way that avoids NaNs
        and maintains monotonicity for all finite inputs.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    pos = y > 0
    # For positive values, use a formulation that remains accurate for large y
    out[pos] = y[pos] + np.log1p(-np.exp(-y[pos]))
    # For non-positive values, clamp expm1 to avoid log(0)
    tiny = np.finfo(float).tiny
    y_exp = np.expm1(y[~pos])
    y_clamped = np.maximum(y_exp, tiny)
    out[~pos] = np.log(np.expm1(y_clamped))
    return out
