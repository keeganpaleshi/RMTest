import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timezone
from typing import Sequence

__all__ = ["to_mpl_times", "setup_time_axis"]

def to_mpl_times(times: Sequence) -> np.ndarray:
    """Convert an array of time values to Matplotlib's numeric format.

    Parameters
    ----------
    times : sequence of float, numpy.datetime64, or datetime.datetime
        Input time values. Floats are interpreted as seconds since the UNIX epoch.

    Returns
    -------
    numpy.ndarray
        Array of floats suitable for plotting with Matplotlib's date functions.
    """
    arr = np.asarray(list(times))
    if arr.size == 0:
        return arr.astype(float)
    if np.issubdtype(arr.dtype, np.datetime64):
        seconds = arr.astype("datetime64[s]").astype(int)
    else:
        first = arr.flat[0]
        if isinstance(first, datetime):
            seconds = np.array(
                [
                    (t if t.tzinfo is None else t.astimezone(timezone.utc))
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    for t in arr
                ],
                dtype=float,
            )
        else:
            seconds = arr.astype(float)
    datetimes = [datetime.utcfromtimestamp(float(s)) for s in seconds]
    return mdates.date2num(datetimes)

def setup_time_axis(ax, times_mpl: np.ndarray) -> None:
    """Configure ``ax`` with UTC dates and elapsed hours secondary axis."""
    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
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

