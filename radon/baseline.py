"""Utilities for subtracting baseline contributions from decay rates."""
import math

__all__ = ["subtract_baseline"]

def subtract_baseline(rate, sigma_rate, baseline_rate, baseline_sigma, scale=1.0):
    """Return baseline-corrected rate and propagated uncertainty.

    Parameters
    ----------
    rate : float
        Measured decay rate (Bq).
    sigma_rate : float
        Uncertainty on ``rate``.
    baseline_rate : float
        Baseline activity in Bq.
    baseline_sigma : float
        Uncertainty on ``baseline_rate``.
    scale : float, optional
        Multiplicative factor applied to ``baseline_rate`` and
        ``baseline_sigma`` before subtraction. Defaults to ``1.0``.

    Returns
    -------
    tuple[float, float]
        ``(corrected_rate, corrected_sigma)``
    """
    rate = float(rate)
    sigma_rate = float(sigma_rate)
    baseline_rate = float(baseline_rate)
    baseline_sigma = float(baseline_sigma)
    corrected = rate - scale * baseline_rate
    error = math.hypot(sigma_rate, scale * baseline_sigma)
    return corrected, error
