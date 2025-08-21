import logging
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils._time_utils import guard_mpl_times, setup_time_axis

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
    times = getattr(ts, "time", None)
    activity = getattr(ts, "activity", None)
    if times is None or activity is None or len(times) == 0 or len(activity) == 0:
        logging.warning("plot_radon_activity: missing data – skipping plot")
        return
    fig, ax = plt.subplots()
    times_mpl = guard_mpl_times(times=times)
    ax.errorbar(times_mpl, activity, yerr=getattr(ts, "error", None), fmt="o")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "radon_activity.png", dpi=300)
    plt.close(fig)


def plot_radon_trend(ts, outdir):
    """Plot radon activity trend without uncertainties."""
    outdir = Path(outdir)
    times = getattr(ts, "time", None)
    activity = getattr(ts, "activity", None)
    if times is None or activity is None or len(times) == 0 or len(activity) == 0:
        logging.warning("plot_radon_trend: missing data – skipping plot")
        return
    fig, ax = plt.subplots()
    times_mpl = guard_mpl_times(times=times)
    ax.plot(times_mpl, activity, "o-")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "radon_trend.png", dpi=300)
    plt.close(fig)
