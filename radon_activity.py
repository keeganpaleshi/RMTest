"""Utilities to combine Po-218 and Po-214 rates into a radon activity."""

import logging
import math
import numpy as np
from typing import Optional, Tuple, cast

__all__ = [
    "clamp_non_negative",
    "compute_radon_activity",
    "compute_total_radon",
    "radon_activity_curve",
    "radon_delta",
    "print_activity_breakdown",
]


def clamp_non_negative(value: float, err: float) -> Tuple[float, float]:
    """Return ``(max(value, 0.0), err)`` and log if clamping occurs."""

    if value < 0:
        logging.warning(
            f"Clamped negative activity (value = {value:.2f} Bq \u2192 0 Bq)"
        )
        value = 0.0
    return value, err


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
        both rates have valid uncertainties.  When only one rate carries a
        valid uncertainty that rate is returned without averaging.  If no
        uncertainties are available the unweighted mean of the supplied
        rates is returned.
    float
        Propagated 1-sigma uncertainty.  When a single rate is used its
        own uncertainty is returned (zero when the rate is treated as
        exact).  When the mean is unweighted the errors are combined as
        ``sqrt(err218**2 + err214**2) / N`` where ``N`` is the number of
        rates included.  Negative uncertainties are ignored while
        zero-valued uncertainties are accepted as exact measurements.  If
        two rates are provided but neither uncertainty is valid, the
        returned uncertainty is ``math.nan``.
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

    values: list[float] = []
    weights: list[Optional[float]] = []

    if rate218 is not None and eff218 > 0:
        if not math.isfinite(rate218):
            logging.warning(
                "Skipping non-finite Po-218 rate when computing radon activity"
            )
        else:
            values.append(rate218)
            if err218 is not None and err218 >= 0:
                if err218 == 0:
                    weights.append(float("inf"))
                elif math.isfinite(err218):
                    weights.append(1.0 / err218**2)
                else:
                    weights.append(None)
            else:
                weights.append(None)

    if rate214 is not None and eff214 > 0:
        if not math.isfinite(rate214):
            logging.warning(
                "Skipping non-finite Po-214 rate when computing radon activity"
            )
        else:
            values.append(rate214)
            if err214 is not None and err214 >= 0:
                if err214 == 0:
                    weights.append(float("inf"))
                elif math.isfinite(err214):
                    weights.append(1.0 / err214**2)
                else:
                    weights.append(None)
            else:
                weights.append(None)

    if not values:
        return 0.0, math.nan

    # If both have valid uncertainties use weighted average
    if len(values) == 2 and all(w is not None for w in weights):
        w1, w2 = cast(Tuple[float, float], (weights[0], weights[1]))
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
            weight_sum = w1 + w2
            if weight_sum <= 0:
                A = (values[0] + values[1]) / 2.0
                sigma = math.nan
            else:
                A = (values[0] * w1 + values[1] * w2) / weight_sum
                sigma = math.sqrt(1.0 / weight_sum)
        return A, sigma

    if len(values) == 2:
        valid_idx = [i for i, w in enumerate(weights) if w is not None]
        if len(valid_idx) == 1:
            idx = valid_idx[0]
            weight = cast(float, weights[idx])
            A = values[idx]
            if math.isinf(weight):
                sigma = 0.0
            else:
                sigma = math.sqrt(1.0 / weight)
            return A, sigma

        A = (values[0] + values[1]) / 2.0
        if all(w is None for w in weights):
            return A, math.nan
        # All remaining cases should correspond to invalid uncertainties for both
        # isotopes; fall back to reporting the combined statistical spread.
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
    *,
    allow_negative_activity: bool = False,
) -> Tuple[float, float, float, float]:
    """Convert activity into concentration and total radon in the sample volume.

    The ``monitor_volume`` of the counting chamber and the ``sample_volume`` of
    the air sample must be supplied using the same units.  The configuration
    files included with this repository use liters, making the derived
    concentration a Bq/L value.

    Both ``monitor_volume`` and ``sample_volume`` must be non-negative.  A
    ``ValueError`` is raised if ``monitor_volume`` is not positive, if
    ``sample_volume`` is negative, or if ``err_bq`` is negative.  Zero
    uncertainties are allowed and treated as exact measurements.  When
    ``activity_bq`` is negative a ``RuntimeError`` is raised unless
    ``allow_negative_activity`` is ``True`` in which case the negative value is
    used without modification.

    Returns
    -------
    concentration : float
        Radon concentration referenced to the combined counting volume of the
        monitor and sampled air (Bq/L when the volumes are given in liters).
        When ``sample_volume`` is zero the concentration is reported per unit of
        the monitor volume because the total and monitor volumes are the same.
    sigma_conc : float
        Uncertainty on the concentration.
    total_bq : float
        Total radon present in the sampled air.  The fitted activity is taken as
        the total activity in the counting chamber and is reported directly.
        When ``sample_volume`` is positive the returned total reflects the
        radon contained in the combined monitor plus sample volume.  For
        background runs with ``sample_volume`` equal to zero, the activity
        measured in the counting chamber is reported instead.
    sigma_total : float
        Uncertainty on ``total_bq``.

    When ``sample_volume`` is zero the total radon reflects only the counting
    chamber contents (no additional scaling is applied) so that background runs
    without a captured air sample still report their measured activity.

    Examples
    --------
    >>> compute_total_radon(5.0, 0.5, 10.0, 20.0)
    (0.16666666666666666, 0.016666666666666666, 5.0, 0.5)
    """
    if monitor_volume <= 0:
        raise ValueError("monitor_volume must be positive")
    if sample_volume < 0:
        raise ValueError("sample_volume must be non-negative")
    if err_bq < 0:
        raise ValueError("err_bq must be non-negative")

    if activity_bq < 0:
        if allow_negative_activity:
            pass
        else:
            clamp_non_negative(activity_bq, err_bq)
            raise RuntimeError(
                "Negative activity encountered. Re-run with --allow_negative_activity to override"
            )
    else:
        activity_bq, err_bq = clamp_non_negative(activity_bq, err_bq)
    if math.isnan(activity_bq):
        raise ValueError("activity_bq must not be NaN")
    total_volume = monitor_volume + sample_volume
    total_bq = activity_bq
    sigma_total = err_bq

    conc = activity_bq / total_volume
    sigma_conc = err_bq / total_volume
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
    half_life_s : float
        Half-life used for the decay model.
    cov_en0 : float, optional
        Covariance between ``E`` and ``N0``. Default is ``0.0`` and disables
        the cross term in the uncertainty propagation.

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


def print_activity_breakdown(rows: list[dict[str, object]]) -> None:
    """Print a table of intermediate radon activity numbers."""

    try:
        from texttable import Texttable  # type: ignore

        table = Texttable()
        table.set_deco(Texttable.BORDER | Texttable.HEADER | Texttable.VLINES)
        table.header(
            [
                "Isotope",
                "Raw (Bq)",
                "Baseline (Bq)",
                "Corrected (Bq)",
                "Err (Bq)",
            ]
        )
        for row in rows:
            table.add_row(
                [
                    row.get("iso", ""),
                    f"{row.get('raw_rate', float('nan')):.3f}",
                    f"{row.get('baseline_rate', float('nan')):.3f}",
                    f"{row.get('corrected', float('nan')):.3f}",
                    f"{row.get('err_corrected', float('nan')):.3f}",
                ]
            )
        print(table.draw())
    except Exception:
        headings = ["Isotope", "Raw (Bq)", "Baseline (Bq)", "Corrected (Bq)", "Err (Bq)"]
        str_rows = [
            [
                str(row.get("iso", "")),
                f"{row.get('raw_rate', float('nan')):.3f}",
                f"{row.get('baseline_rate', float('nan')):.3f}",
                f"{row.get('corrected', float('nan')):.3f}",
                f"{row.get('err_corrected', float('nan')):.3f}",
            ]
            for row in rows
        ]
        widths = [len(h) for h in headings]
        for r in str_rows:
            for i, val in enumerate(r):
                widths[i] = max(widths[i], len(val))

        top = "+" + "+".join("-" * w for w in widths) + "+"
        print(top)
        header = "|" + "|".join(h.ljust(widths[i]) for i, h in enumerate(headings)) + "|"
        print(header)
        middle = "+" + "+".join("-" * w for w in widths) + "+"
        print(middle)
        for r in str_rows:
            line = "|" + "|".join(r[i].ljust(widths[i]) for i in range(len(widths))) + "|"
            print(line)
        bottom = "+" + "+".join("-" * w for w in widths) + "+"
        print(bottom)

    # Compute total radon activity from the corrected rates
    rate218 = err218 = None
    rate214 = err214 = None
    for row in rows:
        if row.get("iso") == "Po218":
            rate218 = float(cast(float, row.get("corrected", 0.0)))
            err218 = float(cast(float, row.get("err_corrected", 0.0)))
        elif row.get("iso") == "Po214":
            rate214 = float(cast(float, row.get("corrected", 0.0)))
            err214 = float(cast(float, row.get("err_corrected", 0.0)))

    total, sigma = compute_radon_activity(rate218, err218, 1.0, rate214, err214, 1.0)
    print(f"Total radon: {total:.3f} +/- {sigma:.3f} Bq")
