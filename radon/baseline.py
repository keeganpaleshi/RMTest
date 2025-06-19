"""Utilities for baseline subtraction."""
import math
from typing import Tuple

def subtract_baseline(counts: float, baseline_counts: float, efficiency: float,
                       live_time: float, baseline_live_time: float) -> Tuple[float, float]:
    """Return background corrected rate and propagated uncertainty.

    Parameters
    ----------
    counts : float
        Total event counts in the analysis window.
    baseline_counts : float
        Counts observed in the baseline period.
    efficiency : float
        Detection efficiency for the isotope.
    live_time : float
        Live time of the analysis measurement in seconds.
    baseline_live_time : float
        Live time of the baseline period in seconds.

    Returns
    -------
    corrected_rate : float
        Baseline corrected decay rate in Bq.
    corrected_sigma : float
        Propagated 1-sigma uncertainty on the corrected rate.
    """
    if efficiency <= 0 or live_time <= 0:
        return 0.0, 0.0
    rate = counts / (live_time * efficiency)
    sigma_rate = math.sqrt(counts) / (live_time * efficiency)
    if baseline_live_time > 0:
        baseline_rate = baseline_counts / (baseline_live_time * efficiency)
        sigma_baseline = math.sqrt(baseline_counts) / (baseline_live_time * efficiency)
    else:
        baseline_rate = 0.0
        sigma_baseline = 0.0
    corrected_rate = rate - baseline_rate
    corrected_sigma = math.hypot(sigma_rate, sigma_baseline)
    return corrected_rate, corrected_sigma
