"""Utility functions for radon baseline subtraction using raw counts."""

import numpy as np

from baseline import _scaling_factor

__all__ = ["subtract_baseline_counts", "subtract_baseline_rate"]


def subtract_baseline_counts(
    counts: float,
    efficiency: float,
    live_time: float,
    baseline_counts: float,
    baseline_live_time: float,
) -> tuple[float, float]:
    """Return background-corrected rate and uncertainty.

    Parameters
    ----------
    counts : float
        Number of signal counts in the analysis window.
    efficiency : float
        Detection efficiency for the signal counts.
    live_time : float
        Live time associated with ``counts`` in seconds.
    baseline_counts : float
        Counts measured in the baseline window.
    baseline_live_time : float
        Live time associated with ``baseline_counts`` in seconds.

    Notes
    -----
    This function operates purely on scalar count values and efficiencies. It is
    **not** intended for DataFrame-based spectra or time series.
    """

    if live_time <= 0:
        raise ValueError("live_time must be positive for baseline correction")
    if baseline_live_time <= 0:
        raise ValueError(
            "baseline_live_time must be positive for baseline correction"
        )
    if efficiency <= 0:
        raise ValueError("efficiency must be positive for baseline correction")

    scale, _ = _scaling_factor(live_time, baseline_live_time)

    net = counts - scale * baseline_counts
    corrected_rate = net / live_time / efficiency

    sigma_sq = counts / live_time**2 / efficiency**2
    baseline_sigma_sq = baseline_counts * scale**2 / live_time**2 / efficiency**2
    corrected_sigma = np.sqrt(sigma_sq + baseline_sigma_sq)
    return corrected_rate, corrected_sigma


def subtract_baseline_rate(
    fit_rate: float,
    fit_sigma: float,
    counts: float,
    efficiency: float,
    live_time: float,
    baseline_counts: float,
    baseline_live_time: float,
    scale: float = 1.0,
) -> tuple[float, float, float, float]:
    """Apply baseline subtraction to a fitted decay rate.

    Parameters
    ----------
    fit_rate : float
        Rate from the time-series fit in Bq.
    fit_sigma : float
        Uncertainty on ``fit_rate`` in Bq.
    counts : float
        Unweighted counts from the analysis window used for the fit.
    efficiency : float
        Detection efficiency for the isotope.
    live_time : float
        Live time associated with ``counts`` in seconds.
    baseline_counts : float
        Counts measured in the baseline window.
    baseline_live_time : float
        Live time of the baseline window in seconds.
    scale : float, optional
        Additional multiplicative scale applied to the baseline rate,
        default is ``1.0``.

    Returns
    -------
    corrected_rate : float
        Baseline-subtracted rate in Bq.
    corrected_sigma : float
        Combined 1-sigma uncertainty.
    baseline_rate : float
        Raw baseline rate in Bq before scaling.
    baseline_sigma : float
        Statistical uncertainty on ``baseline_rate``.
    """

    if live_time <= 0:
        raise ValueError("live_time must be positive for baseline correction")
    if baseline_live_time <= 0:
        raise ValueError(
            "baseline_live_time must be positive for baseline correction"
        )
    if efficiency <= 0:
        raise ValueError("efficiency must be positive for baseline correction")

    baseline_rate = baseline_counts / (baseline_live_time * efficiency)
    baseline_sigma = np.sqrt(baseline_counts) / (baseline_live_time * efficiency)

    _, sigma_rate = subtract_baseline_counts(
        counts,
        efficiency,
        live_time,
        baseline_counts,
        baseline_live_time,
    )

    corrected_rate = fit_rate - scale * baseline_rate
    corrected_sigma = float(np.hypot(fit_sigma, sigma_rate * scale))

    return corrected_rate, corrected_sigma, baseline_rate, baseline_sigma
