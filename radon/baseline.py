import numpy as np

__all__ = ["subtract_baseline_counts"]


def subtract_baseline_counts(
    counts: float,
    efficiency: float,
    live_time: float,
    baseline_counts: float,
    baseline_live_time: float,
) -> tuple[float, float]:
    """Return baseline-corrected rate and uncertainty.

    Parameters
    ----------
    counts : float
        Total counts in the signal region.
    efficiency : float
        Detection efficiency of the signal.
    live_time : float
        Exposure time for ``counts`` in seconds.
    baseline_counts : float
        Counts measured during the baseline interval.
    baseline_live_time : float
        Exposure time for the baseline counts in seconds.

    Returns
    -------
    tuple of float
        Corrected rate in counts per second and its one-sigma uncertainty.
    """

    rate = counts / live_time / efficiency
    sigma_sq = counts / live_time**2 / efficiency**2
    baseline_rate = baseline_counts / baseline_live_time / efficiency
    baseline_sigma_sq = baseline_counts / baseline_live_time**2 / efficiency**2
    corrected_rate = rate - baseline_rate
    corrected_sigma = np.sqrt(sigma_sq + baseline_sigma_sq)
    return corrected_rate, corrected_sigma
