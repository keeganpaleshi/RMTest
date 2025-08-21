import numpy as np


def log_expm1_stable(y: np.ndarray) -> np.ndarray:
    """Compute log(expm1(y)) in a numerically stable manner.

    Parameters
    ----------
    y : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        ``log(expm1(y))`` evaluated elementwise with safeguards against
        overflow and ``log(0)``.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask = y > 0
    out[mask] = y[mask] + np.log1p(-np.exp(-y[mask]))
    tiny = np.finfo(float).tiny
    exm1 = np.expm1(y[~mask])
    exm1 = np.clip(exm1, tiny, np.inf)
    out[~mask] = np.log(exm1)
    return out
