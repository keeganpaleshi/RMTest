import numpy as np

__all__ = ["_scaling_factor"]


def _scaling_factor(dt_window: float, dt_baseline: float,
                    err_window: float = 0.0,
                    err_baseline: float = 0.0) -> tuple[float, float]:
    """Return scaling factor between analysis and baseline durations.

    This helper computes ``dt_window / dt_baseline`` and propagates the
    1-sigma uncertainty from ``err_window`` and ``err_baseline`` assuming they
    are independent.  A ``ValueError`` is raised when ``dt_baseline`` is zero
    to avoid division by zero.
    """

    if dt_baseline == 0:
        raise ValueError("dt_baseline must be non-zero")

    scale = float(dt_window) / float(dt_baseline)
    var = (err_window / dt_baseline) ** 2
    var += ((dt_window * err_baseline) / dt_baseline**2) ** 2
    return scale, float(np.sqrt(var))
