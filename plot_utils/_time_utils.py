"""Internal helpers for time axis formatting and conversions."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np


def to_mpl_times(times: Iterable) -> np.ndarray:
    """Convert an array-like of times to Matplotlib date numbers.

    Parameters
    ----------
    times : Iterable
        Sequence of epoch seconds, :class:`numpy.datetime64`, or
        :class:`datetime.datetime` objects.

    Returns
    -------
    np.ndarray
        Array of floats suitable for Matplotlib time plotting.
    """

    arr = np.asarray(list(times))
    if np.issubdtype(arr.dtype, np.datetime64):
        secs = arr.astype("datetime64[s]").astype(np.int64).astype(float)
    elif arr.dtype == object:
        secs_list: list[float] = []
        for t in arr:
            if isinstance(t, datetime):
                secs_list.append(t.timestamp())
            elif isinstance(t, np.datetime64):
                secs_list.append(float(t.astype("datetime64[s]").astype(np.int64)))
            else:
                secs_list.append(float(t))
        secs = np.array(secs_list, dtype=float)
    else:
        secs = arr.astype(float)
    # mdates.epoch2num is not available in older Matplotlib versions,
    # so perform the conversion manually: seconds to days since 1970-01-01.
    epoch = mdates.date2num(datetime(1970, 1, 1))
    return secs / 86400.0 + epoch


def setup_time_axis(ax, times_mpl: np.ndarray):
    """Apply UTC date labels and elapsed-hour secondary axis."""
    locator = mdates.AutoDateLocator()
    try:  # Concise formatter is available on newer Matplotlib
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:  # pragma: no cover - fallback for old MPL
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    base = times_mpl[0]

    def _to_hours(x):
        return (x - base) * 24.0

    def _to_dates(x):
        return base + x / 24.0

    secax = ax.secondary_xaxis("top", functions=(_to_hours, _to_dates))
    secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos=None: f"{x:g}"))
    secax.set_xlabel("Elapsed Time (h)")

    ax.xaxis.get_offset_text().set_visible(False)
    secax.xaxis.get_offset_text().set_visible(False)
    return secax
