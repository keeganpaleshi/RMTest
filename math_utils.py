import numpy as np


def log_expm1_stable(y: np.ndarray) -> np.ndarray:
    """Compute log(expm1(y)) robustly for all finite y.

    Parameters
    ----------
    y : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        ``log(expm1(y))`` evaluated elementwise with overflow protection.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask = y > 0
    # For positive values use stable formulation that avoids overflow
    out[mask] = y[mask] + np.log1p(-np.exp(-y[mask]))
    # For non-positive values fallback to direct computation while
    # clamping the argument to avoid log(0)
    tiny = np.finfo(float).tiny
    tmp = np.expm1(y[~mask])
    tmp = np.clip(tmp, tiny, None)
    out[~mask] = np.log(tmp)
    return out
