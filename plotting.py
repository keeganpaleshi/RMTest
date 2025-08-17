import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime
from pathlib import Path

__all__ = ["plot_radon_activity", "plot_radon_trend"]


def _format_time_axes(ax, times_dt):
    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)

    t0 = times_dt[0]

    def to_hours(x):
        return (x - t0) * 24.0

    def to_dates(h):
        return t0 + h / 24.0

    secax = ax.secondary_xaxis("top", functions=(to_hours, to_dates))
    secax.set_xlabel("Elapsed Time (h)")
    secax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    secax.xaxis.get_offset_text().set_visible(False)
    return secax


def plot_radon_activity(ts, outdir):
    """Plot radon activity with uncertainties.

    Parameters
    ----------
    ts : object
        Object with ``time``, ``activity`` and ``error`` attributes.
    outdir : Path or str
        Output directory where ``radon_activity.png`` will be saved.
    """
    outdir = Path(outdir)
    fig, ax = plt.subplots()
    times_dt = mdates.date2num([datetime.utcfromtimestamp(t) for t in ts.time])
    ax.errorbar(times_dt, ts.activity, yerr=getattr(ts, "error", None), fmt="o")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    _format_time_axes(ax, times_dt)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "radon_activity.png", dpi=300)
    plt.close(fig)


def plot_radon_trend(ts, outdir):
    """Plot radon activity trend without uncertainties."""
    outdir = Path(outdir)
    fig, ax = plt.subplots()
    times_dt = mdates.date2num([datetime.utcfromtimestamp(t) for t in ts.time])
    ax.plot(times_dt, ts.activity, "o-")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    _format_time_axes(ax, times_dt)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "radon_trend.png", dpi=300)
    plt.close(fig)
