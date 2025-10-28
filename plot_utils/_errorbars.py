"""Helpers for plotting routines that include uncertainty bands."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def apply_error_limits(
    ax: Axes,
    values,
    errors=None,
    *,
    pad_fraction: float = 0.1,
) -> None:
    """Adjust ``ax`` y-limits to encompass values and their uncertainties.

    Parameters
    ----------
    ax:
        Matplotlib axes whose limits will be updated in-place.
    values:
        Sequence of y-values used in the plot.
    errors:
        Optional sequence of symmetric 1-sigma uncertainties for the
        ``values``.  When provided and broadcastable to ``values`` the upper
        and lower envelopes are used to determine the axis limits.
    pad_fraction:
        Fractional padding applied to the computed span to avoid clipping the
        extremal points.  Defaults to ``0.1`` (10%).
    """

    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return

    lower = vals.copy()
    upper = vals.copy()

    if errors is not None:
        errs = np.asarray(errors, dtype=float)
        if errs.size == vals.size:
            lower = vals - errs
            upper = vals + errs

    finite = np.isfinite(lower) & np.isfinite(upper)
    if not np.any(finite):
        return

    y_min = float(np.min(lower[finite]))
    y_max = float(np.max(upper[finite]))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return

    if y_min == y_max:
        span = abs(y_min) if y_min != 0.0 else 1.0
        padding = span * pad_fraction
        y_min -= padding
        y_max += padding
    else:
        span = y_max - y_min
        padding = span * pad_fraction if span > 0 else max(abs(y_min), abs(y_max), 1.0) * pad_fraction
        y_min -= padding
        y_max += padding

    ax.set_ylim(y_min, y_max)

