"""Utility functions for radon baseline subtraction using raw counts."""

import numpy as np

__all__ = ["subtract_baseline_counts"]


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

    if live_time <= 0 or efficiency <= 0:
        return 0.0, 0.0
    rate = counts / live_time / efficiency
    sigma_sq = counts / live_time**2 / efficiency**2
    if baseline_live_time > 0:
        baseline_rate = baseline_counts / baseline_live_time / efficiency
        baseline_sigma_sq = baseline_counts / baseline_live_time**2 / efficiency**2
    else:
        baseline_rate = 0.0
        baseline_sigma_sq = 0.0
    corrected_rate = rate - baseline_rate
    corrected_sigma = np.sqrt(sigma_sq + baseline_sigma_sq)
    return corrected_rate, corrected_sigma
