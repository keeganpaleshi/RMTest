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
    # For very small positive values, expm1(y) ~= y and the generic
    # formulation suffers catastrophic cancellation via log1p(-exp(-y)).
    small_mask = mask & (y < 1.0e-8)
    if np.any(small_mask):
        out[small_mask] = np.log(np.maximum(y[small_mask], np.finfo(float).tiny))
    mid_mask = mask & ~small_mask
    # For positive values use stable formulation that avoids overflow.
    if np.any(mid_mask):
        out[mid_mask] = y[mid_mask] + np.log1p(-np.exp(-y[mid_mask]))
    # For non-positive inputs defer to NumPy's behaviour so that values
    # outside the domain ``expm1(y) > 0`` propagate ``nan``/``-inf``
    # instead of being silently clamped to tiny positives.
    out[~mask] = np.log(np.expm1(y[~mask]))
    return out
