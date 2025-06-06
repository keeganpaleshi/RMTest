"""Utilities to combine Po-218 and Po-214 rates into a radon activity.
"""

import math
from typing import Optional, Tuple

__all__ = [
    "compute_radon_activity",
    "compute_total_radon",
    "radon_activity_curve",
]


def compute_radon_activity(
    rate218: Optional[float] = None,
    err218: Optional[float] = None,
    eff218: float = 1.0,
    rate214: Optional[float] = None,
    err214: Optional[float] = None,
    eff214: float = 1.0,
) -> Tuple[float, float]:
    """Combine Po-218 and Po-214 rates into a radon activity.

    Parameters
    ----------
    rate218, rate214 : float or None
        Measured count rates for the two isotopes.
    err218, err214 : float or None
        Uncertainties on the rates.
    eff218, eff214 : float
        Detection efficiencies used to convert counts to Bq.

    Returns
    -------
    float
        Weighted average radon activity in Bq.
    float
        Propagated 1-sigma uncertainty.
    """
    values = []
    weights = []

    if rate218 is not None:
        val = rate218 / eff218 if eff218 > 0 else 0.0
        values.append(val)
        if err218 is not None and err218 > 0 and eff218 > 0:
            weights.append(1.0 / (err218 / eff218) ** 2)
        else:
            weights.append(None)

    if rate214 is not None:
        val = rate214 / eff214 if eff214 > 0 else 0.0
        values.append(val)
        if err214 is not None and err214 > 0 and eff214 > 0:
            weights.append(1.0 / (err214 / eff214) ** 2)
        else:
            weights.append(None)

    if not values:
        return 0.0, 0.0

    # If both have valid uncertainties use weighted average
    if len(values) == 2 and all(w is not None for w in weights):
        w1, w2 = weights
        A = (values[0] * w1 + values[1] * w2) / (w1 + w2)
        sigma = math.sqrt(1.0 / (w1 + w2))
        return A, sigma

    if len(values) == 2 and sum(w is not None for w in weights) == 1:
        # Identify the isotope with a valid uncertainty
        valid_idx = 0 if weights[0] is not None else 1
        return values[valid_idx], math.sqrt(1.0 / weights[valid_idx])

    # Only one valid value or missing errors
    A = values[0]
    sigma = math.sqrt(1.0 / weights[0]) if weights[0] is not None else 0.0
    return A, sigma


def compute_total_radon(
    activity_bq: float,
    err_bq: float,
    monitor_volume: float,
    sample_volume: float,
) -> Tuple[float, float, float, float]:
    """Convert activity into concentration and total radon in the sample volume.

    Returns
    -------
    concentration : float
        Radon concentration in Bq per same unit as ``monitor_volume``.
    sigma_conc : float
        Uncertainty on the concentration.
    total_bq : float
        Total radon in the sample volume in Bq.
    sigma_total : float
        Uncertainty on ``total_bq``.
    """
    if monitor_volume <= 0:
        raise ValueError("monitor_volume must be positive")
    conc = activity_bq / monitor_volume
    sigma_conc = err_bq / monitor_volume

    total_bq = conc * sample_volume
    sigma_total = sigma_conc * sample_volume
    return conc, sigma_conc, total_bq, sigma_total


def radon_activity_curve(
    times,
    E: float,
    dE: float,
    N0: float,
    dN0: float,
    half_life_s: float,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Activity over time from fitted decay parameters.

    Parameters
    ----------
    times : array-like
        Relative times in seconds.
    E : float
        Steady-state decay rate in Bq.
    dE : float
        Uncertainty on ``E``.
    N0 : float
        Initial activity parameter.
    dN0 : float
        Uncertainty on ``N0``.
    half_life_s : float
        Half-life used for the decay model.

    Returns
    -------
    numpy.ndarray
        Activity at each time in Bq.
    numpy.ndarray
        Propagated 1-sigma uncertainty at each time.
    """
    import numpy as np

    t = np.asarray(times, dtype=float)
    lam = math.log(2.0) / float(half_life_s)
    exp_term = np.exp(-lam * t)
    activity = E * (1.0 - exp_term) + lam * N0 * exp_term

    dA_dE = 1.0 - exp_term
    dA_dN0 = lam * exp_term
    variance = (dA_dE * dE) ** 2 + (dA_dN0 * dN0) ** 2
    sigma = np.sqrt(variance)
    return activity, sigma
