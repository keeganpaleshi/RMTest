import numpy as np


def log_expm1_stable(y: np.ndarray) -> np.ndarray:
    """Stable computation of log(expm1(y)) for all finite y.

    Parameters
    ----------
    y : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        log(expm1(y)) evaluated in a numerically stable manner.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y, dtype=float)
    pos = y > 0
    out[pos] = y[pos] + np.log1p(-np.exp(-y[pos]))
    neg = ~pos
    if np.any(neg):
        tiny = np.finfo(float).tiny
        tmp = np.expm1(y[neg])
        tmp = np.clip(tmp, tiny, np.inf)
        out[neg] = np.log(tmp)
    return out
