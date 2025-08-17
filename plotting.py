import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime
from pathlib import Path

__all__ = ["plot_radon_activity", "plot_radon_trend"]


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
    times = [datetime.utcfromtimestamp(t) for t in ts.time]
    times_dt = mdates.date2num(times)
    ax.errorbar(times_dt, ts.activity, yerr=getattr(ts, "error", None), fmt="o")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")

    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    base_dt = times_dt[0]

    def _to_hours(x):
        return (x - base_dt) * 24.0

    def _to_dates(x):
        return base_dt + x / 24.0

    secax = ax.secondary_xaxis("top", functions=(_to_hours, _to_dates))
    secax.set_xlabel("Elapsed Time (h)")
    secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))

    ax.xaxis.get_offset_text().set_visible(False)
    secax.xaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "radon_activity.png", dpi=300)
    plt.close(fig)


def plot_radon_trend(ts, outdir):
    """Plot radon activity trend without uncertainties."""
    outdir = Path(outdir)
    fig, ax = plt.subplots()
    times = [datetime.utcfromtimestamp(t) for t in ts.time]
    times_dt = mdates.date2num(times)
    ax.plot(times_dt, ts.activity, "o-")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")

    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    base_dt = times_dt[0]

    def _to_hours(x):
        return (x - base_dt) * 24.0

    def _to_dates(x):
        return base_dt + x / 24.0

    secax = ax.secondary_xaxis("top", functions=(_to_hours, _to_dates))
    secax.set_xlabel("Elapsed Time (h)")
    secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))

    ax.xaxis.get_offset_text().set_visible(False)
    secax.xaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "radon_trend.png", dpi=300)
    plt.close(fig)
