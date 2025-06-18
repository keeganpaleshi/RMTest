# -----------------------------------------------------
# efficiency.py
# -----------------------------------------------------
"""Detection efficiency calculations and BLUE combination."""

from typing import Sequence, Tuple, Optional
import numpy as np
import math
import warnings

__all__ = [
    "calc_spike_efficiency",
    "calc_assay_efficiency",
    "calc_decay_efficiency",
    "blue_combine",
]


def calc_spike_efficiency(
    counts: float, activity_bq: float, live_time_s: float
) -> float:
    """Compute spike efficiency from counts and known activity.

    Parameters
    ----------
    counts : float
        Number of counts observed during the spike run.
    activity_bq : float
        Known spike activity in Bq.
    live_time_s : float
        Exposure time in seconds.

    Returns
    -------
    float
        Detection efficiency ``counts / (activity_bq * live_time_s)``.
    """
    if activity_bq <= 0 or live_time_s <= 0:
        raise ValueError("activity_bq and live_time_s must be positive")
    if counts < 0:
        raise ValueError("counts cannot be negative")
    return float(counts) / (float(activity_bq) * float(live_time_s))


def calc_assay_efficiency(rate_cps: float, reference_bq: float) -> float:
    """Calculate efficiency from assay measurement.

    Parameters
    ----------
    rate_cps : float
        Measured count rate in counts/s.
    reference_bq : float
        Reference activity in Bq.

    Returns
    -------
    float
        Efficiency ``rate_cps / reference_bq``.
    """
    if reference_bq <= 0:
        raise ValueError("reference_bq must be positive")
    if rate_cps < 0:
        raise ValueError("rate_cps cannot be negative")
    return float(rate_cps) / float(reference_bq)


def calc_decay_efficiency(observed_rate_cps: float, expected_rate_cps: float) -> float:
    """Efficiency derived from a decay curve.

    Parameters
    ----------
    observed_rate_cps : float
        Fitted decay rate from the measurement.
    expected_rate_cps : float
        True decay rate expected from known activity.

    Returns
    -------
    float
        Efficiency ``observed_rate_cps / expected_rate_cps``.
    """
    if expected_rate_cps <= 0:
        raise ValueError("expected_rate_cps must be positive")
    if observed_rate_cps < 0:
        raise ValueError("observed_rate_cps cannot be negative")
    return float(observed_rate_cps) / float(expected_rate_cps)


def blue_combine(
    values: Sequence[float],
    errors: Sequence[float],
    corr: Optional[np.ndarray] = None,
    *,
    allow_negative: bool = False,
) -> Tuple[float, float, np.ndarray]:
    """Combine estimates using the BLUE method.

    Parameters
    ----------
    values : sequence of float
        Individual efficiency estimates.
    errors : sequence of float
        1-sigma uncertainties associated with ``values``.
    corr : array-like, optional
        Correlation matrix.  If ``None`` the estimates are assumed
        uncorrelated.

    Returns
    -------
    float
        Combined estimate.
    float
        Combined 1-sigma uncertainty.
    numpy.ndarray
        Weights applied to the input values.

    Notes
    -----
    If ``allow_negative`` is ``False`` and any resulting weight is
    negative, a :class:`ValueError` is raised.  Otherwise a
    ``UserWarning`` is emitted.
    """
    vals = np.asarray(values, dtype=float)
    errs = np.asarray(errors, dtype=float)
    if vals.size != errs.size:
        raise ValueError("values and errors must have the same length")
    if vals.size == 0:
        raise ValueError("no values provided")

    if corr is None:
        cov = np.diag(errs**2)
    else:
        c = np.asarray(corr, dtype=float)
        if c.shape != (vals.size, vals.size):
            raise ValueError("correlation matrix has wrong shape")
        cov = c * np.outer(errs, errs)

    Vinv = np.linalg.inv(cov)
    ones = np.ones(vals.size)
    norm = float(ones @ Vinv @ ones)
    weights = (Vinv @ ones) / norm
    if np.any(weights < 0):
        if allow_negative:
            warnings.warn("negative BLUE weights encountered")
        else:
            raise ValueError("negative BLUE weights encountered")
    weights /= weights.sum()
    estimate = float(weights @ vals)
    variance = 1.0 / norm
    return estimate, math.sqrt(variance), weights
