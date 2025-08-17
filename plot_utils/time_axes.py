import matplotlib.dates as mdates
import matplotlib.ticker as mticker


def configure_time_axes(ax, times_dt, *, label="Elapsed Time (h)"):
    """Format bottom axis as datetimes and add elapsed-hours on top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to format.
    times_dt : array-like
        Times expressed as Matplotlib datetimes (days since epoch).
    label : str, optional
        Label to use for the secondary elapsed-time axis.
    """
    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:  # pragma: no cover - old matplotlib
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    base = times_dt[0] if len(times_dt) else 0.0

    def _to_hours(x):
        return (x - base) * 24.0

    def _to_dates(h):
        return base + h / 24.0

    secax = ax.secondary_xaxis("top", functions=(_to_hours, _to_dates))
    secax.set_xlabel(label)
    secax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda h, _pos: f"{h:g}")
    )

    ax.xaxis.get_offset_text().set_visible(False)
    secax.xaxis.get_offset_text().set_visible(False)
    return secax
