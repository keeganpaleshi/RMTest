from baseline_utils import rate_histogram, subtract


def calc_activity(counts: float, live_time_s: float) -> float:
    """Compute activity from event counts and exposure time.

    Parameters
    ----------
    counts : float
        Number of detected counts.
    live_time_s : float
        Exposure time in seconds.

    Returns
    -------
    float
        Activity in Bq.

    Raises
    ------
    ValueError
        If ``live_time_s`` is non-positive or ``counts`` is negative.
    """

    if live_time_s <= 0:
        raise ValueError("live_time_s must be positive")
    if counts < 0:
        raise ValueError("counts cannot be negative")
    return float(counts) / float(live_time_s)

# Backwards compatibility
subtract_baseline = subtract

__all__ = ["rate_histogram", "subtract", "subtract_baseline", "calc_activity"]
