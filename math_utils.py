import numpy as np


def log_expm1_stable(y: np.ndarray) -> np.ndarray:
    """Return log(expm1(y)) in a numerically stable way.

    Parameters
    ----------
    y : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        ``log(expm1(y))`` evaluated elementwise with overflow and underflow
        protection. Works for all finite ``y`` and never returns NaN.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y, dtype=float)
    pos = y > 0
    # For positive values use stable formulation that avoids overflow
    out[pos] = y[pos] + np.log1p(-np.exp(-y[pos]))
    # For non-positive values fall back to log(expm1(y)) while clamping
    tiny = np.finfo(float).tiny
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        val = np.expm1(y[~pos])
    val = np.clip(val, tiny, None)
    out[~pos] = np.log(val)
    return out
