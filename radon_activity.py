"""Utilities to combine Po-218 and Po-214 rates into a radon activity."""

import math
import numpy as np
from typing import Optional, Tuple

__all__ = [
    "compute_radon_activity",
    "compute_total_radon",
    "radon_activity_curve",
    "radon_delta",
]


def compute_radon_activity(
    rate218: Optional[float] = None,
    err218: Optional[float] = None,
    eff218: float = 1.0,
    rate214: Optional[float] = None,
    err214: Optional[float] = None,
    eff214: float = 1.0,
    *,
    t_since_start: Optional[float] = None,
    settle_time: float = 0.0,
    require_equilibrium: bool = True,
) -> Tuple[float, float]:
    """Combine Po-218 and Po-214 rates into a radon activity.

    The input decay rates should generally be measured after the
    short-lived daughters have reached secular equilibrium with radon.
    Using early-time rates can bias the result unless the discrepancy is
    intentional.  When ``require_equilibrium`` is ``True`` this function
    can enforce the condition by comparing ``t_since_start`` with
    ``settle_time``.

    Parameters
    ----------
    rate218, rate214 : float or None
        Measured activities for the two isotopes in Bq.  The values should
        already include the detection efficiencies; this function will not
        scale them further.
    err218, err214 : float or None
        Uncertainties on the rates in Bq.  Negative values are ignored
        while zero values are accepted and treated as exact measurements.
    eff218, eff214 : float
        Detection efficiencies for the two isotopes.  They are only used to
        determine whether an isotope contributes to the average: a value of
        zero disables that isotope and negative values raise a ``ValueError``.
        The efficiencies are not applied as multiplicative weights.
    t_since_start : float, optional
        Time in seconds since counting began. Only checked when
        ``require_equilibrium`` is ``True``.
    settle_time : float, optional
        Minimum time in seconds required for secular equilibrium.
        Defaults to ``0.0``.
    require_equilibrium : bool, optional
        When ``True`` and ``t_since_start`` is provided, raise a
        ``ValueError`` if it is smaller than ``settle_time``.

    Returns
    -------
    float
        Average radon activity in Bq.  A weighted mean is used only when
        both rates have valid uncertainties; otherwise an unweighted mean
        of the available rates is returned.
    float
        Propagated 1-sigma uncertainty.  When the mean is unweighted the
        errors are combined as ``sqrt(err218**2 + err214**2) / N`` where
        ``N`` is the number of rates included.  Negative uncertainties are
        ignored while zero-valued uncertainties are accepted as exact
        measurements.  If two rates are provided but neither uncertainty is
        valid, the returned uncertainty is ``math.nan``.
    """
    if eff218 < 0:
        raise ValueError("eff218 must be non-negative")
    if eff214 < 0:
        raise ValueError("eff214 must be non-negative")
    if require_equilibrium and t_since_start is not None:
        if t_since_start < settle_time:
            raise ValueError(
                "rates must be measured after secular equilibrium or set "
                "require_equilibrium=False"
            )

    values = []
    weights = []

    if rate218 is not None and eff218 > 0:
        values.append(rate218)
        if err218 is not None and err218 >= 0:
            if err218 == 0:
                weights.append(float("inf"))
            else:
                weights.append(1.0 / err218**2)
        else:
            weights.append(None)

    if rate214 is not None and eff214 > 0:
        values.append(rate214)
        if err214 is not None and err214 >= 0:
            if err214 == 0:
                weights.append(float("inf"))
            else:
                weights.append(1.0 / err214**2)
        else:
            weights.append(None)

    if not values:
        return 0.0, math.nan

    # If both have valid uncertainties use weighted average
    if len(values) == 2 and all(w is not None for w in weights):
        w1, w2 = weights
        if math.isinf(w1) and math.isinf(w2):
            A = (values[0] + values[1]) / 2.0
            sigma = 0.0
        elif math.isinf(w1):
            A = values[0]
            sigma = 0.0
        elif math.isinf(w2):
            A = values[1]
            sigma = 0.0
        else:
            A = (values[0] * w1 + values[1] * w2) / (w1 + w2)
            sigma = math.sqrt(1.0 / (w1 + w2))
        return A, sigma

    if len(values) == 2:
        A = (values[0] + values[1]) / 2.0
        if all(w is None for w in weights):
            return A, math.nan
        # Unweighted average when any uncertainty is missing or invalid
        e218 = err218 if err218 is not None and err218 >= 0 else 0.0
        e214 = err214 if err214 is not None and err214 >= 0 else 0.0
        sigma = math.sqrt(e218**2 + e214**2) / 2.0
        return A, sigma

    # Only one valid value or missing errors
    A = values[0]
    if weights[0] is None:
        sigma = math.nan
    elif math.isinf(weights[0]):
        sigma = 0.0
    else:
        sigma = math.sqrt(1.0 / weights[0])
    return A, sigma


def compute_total_radon(
    activity_bq: float,
    err_bq: float,
    monitor_volume: float,
    sample_volume: float,
) -> Tuple[float, float, float, float]:
    """Convert activity into concentration and total radon in the sample volume.

    The ``monitor_volume`` of the counting chamber and the ``sample_volume`` of
    the air sample must be supplied using the same units.  The configuration
    files included with this repository use liters.

    Both ``monitor_volume`` and ``sample_volume`` must be non-negative.  A
    ``ValueError`` is raised if ``monitor_volume`` is not positive, if
    ``sample_volume`` is negative, if ``err_bq`` is negative, or if
    ``activity_bq`` is negative.

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

    Examples
    --------
    >>> compute_total_radon(5.0, 0.5, 10.0, 20.0)
    (0.5, 0.05, 10.0, 1.0)
    """
    if monitor_volume <= 0:
        raise ValueError("monitor_volume must be positive")
    if sample_volume < 0:
        raise ValueError("sample_volume must be non-negative")
    if err_bq < 0:
        raise ValueError("err_bq must be non-negative")
    if activity_bq < 0:
        raise ValueError("activity_bq must be non-negative")
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
    cov_en0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
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
    cov_en0 : float, optional
        Covariance between ``E`` and ``N0``. Default is ``0.0`` and disables
        the cross term in the uncertainty propagation.
    half_life_s : float
        Half-life used for the decay model.

    Returns
    -------
    numpy.ndarray
        Activity at each time in Bq.
    numpy.ndarray
        Propagated 1-sigma uncertainty at each time.
    """

    if half_life_s <= 0:
        raise ValueError("half_life_s must be positive")

    t = np.asarray(times, dtype=float)
    lam = math.log(2.0) / float(half_life_s)
    exp_term = np.exp(-lam * t)
    activity = E * (1.0 - exp_term) + lam * N0 * exp_term

    dA_dE = 1.0 - exp_term
    dA_dN0 = lam * exp_term
    variance = (dA_dE * dE) ** 2 + (dA_dN0 * dN0) ** 2
    if cov_en0:
        variance += 2.0 * dA_dE * dA_dN0 * cov_en0
    sigma = np.sqrt(variance)
    return activity, sigma


def radon_delta(
    t_start: float,
    t_end: float,
    E: float,
    dE: float,
    N0: float,
    dN0: float,
    half_life_s: float,
    cov_en0: float = 0.0,
) -> Tuple[float, float]:
    """Change in activity between two times.

    Parameters are identical to :func:`radon_activity_curve` with ``t_start``
    and ``t_end`` specifying the relative times in seconds. ``cov_en0`` is the
    optional covariance between ``E`` and ``N0`` used for the uncertainty
    propagation.
    """

    if half_life_s <= 0:
        raise ValueError("half_life_s must be positive")

    lam = math.log(2.0) / float(half_life_s)
    exp1 = math.exp(-lam * float(t_start))
    exp2 = math.exp(-lam * float(t_end))

    delta = E * (exp1 - exp2) + lam * N0 * (exp2 - exp1)

    d_delta_dE = exp1 - exp2
    d_delta_dN0 = lam * (exp2 - exp1)
    variance = (d_delta_dE * dE) ** 2 + (d_delta_dN0 * dN0) ** 2
    if cov_en0:
        variance += 2.0 * d_delta_dE * d_delta_dN0 * cov_en0
    sigma = math.sqrt(variance)
    return delta, sigma
