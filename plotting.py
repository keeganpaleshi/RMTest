import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from plot_utils._time_utils import guard_mpl_times, setup_time_axis

__all__ = ["plot_radon_activity", "plot_radon_trend", "plot_time_series"]


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
    times_mpl = guard_mpl_times(times=ts.time)
    ax.errorbar(times_mpl, ts.activity, yerr=getattr(ts, "error", None), fmt="o")
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
    fig, ax = plt.subplots()
    times_mpl = guard_mpl_times(times=ts.time)
    ax.plot(times_mpl, ts.activity, "o-")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "radon_trend.png", dpi=300)
    plt.close(fig)


def plot_time_series(series_map, outdir):
    """Plot time series for Po214 and Po210 with guards for missing data."""

    outdir = Path(outdir)
    isotopes = ["Po214", "Po210"]
    valid = []
    for iso in isotopes:
        data = series_map.get(iso)
        if not data or data.get("time") is None or len(data.get("time")) == 0:
            logging.warning("No data for %s – skipping subplot", iso)
            continue
        valid.append((iso, data))

    if not valid:
        logging.warning("No time-series data to plot")
        return

    fig, axes = plt.subplots(len(valid), 1, sharex=True)
    if len(valid) == 1:
        axes = [axes]

    for ax, (iso, data) in zip(axes, valid):
        times_mpl = guard_mpl_times(times=data["time"])
        ax.plot(times_mpl, data["activity"], "o-")
        ax.set_ylabel(f"{iso} rate [Bq]")
        setup_time_axis(ax, times_mpl)

    axes[-1].set_xlabel("Time (UTC)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "time_series.png", dpi=300)
    plt.close(fig)
