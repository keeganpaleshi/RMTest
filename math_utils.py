import numpy as np


def log_expm1_stable(x):
    """Return ``log(exp(x) - 1)`` in a numerically stable way.

    For large positive ``x`` values ``np.expm1(x)`` will overflow. In this
    regime we instead compute ``x + log1p(-exp(-x))`` which is algebraically
    equivalent but avoids overflow. For smaller values ``np.log(np.expm1(x))``
    provides accurate results.

    Parameters
    ----------
    x : array_like
        Input value or array of values.

    Returns
    -------
    numpy.ndarray or float
        The element-wise ``log(exp(x) - 1)``.
    """
    x_arr = np.asarray(x, dtype=float)
    threshold = np.log(np.finfo(float).max)

    # ``np.where`` evaluates both branches so we must explicitly mask to avoid
    # calling ``np.expm1`` on large values which would overflow.  Working on the
    # masked arrays keeps the implementation vectorised while preventing
    # spurious runtime warnings.
    res = np.empty_like(x_arr)
    mask = x_arr < threshold
    res[mask] = np.log(np.expm1(x_arr[mask]))
    res[~mask] = x_arr[~mask] + np.log1p(-np.exp(-x_arr[~mask]))
    return res
