"""Radon plotting helpers with standardized time axes."""

import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime
from pathlib import Path


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


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(
    ts_dict, outdir: Path, out_png: str | Path | None = None
) -> None:
    t = np.asarray(ts_dict["time"], dtype=float)
    a = np.asarray(ts_dict["activity"], dtype=float)
    e = np.asarray(ts_dict["error"], dtype=float)
    times_dt = mdates.date2num([datetime.utcfromtimestamp(val) for val in t])
    fig, ax = plt.subplots()
    ax.errorbar(times_dt, a, yerr=e, fmt="o")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    _format_time_axes(ax, times_dt)
    fig.autofmt_xdate()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_activity")


def plot_radon_trend(
    ts_dict, outdir: Path, out_png: str | Path | None = None
) -> None:
    t = np.asarray(ts_dict["time"], dtype=float)
    a = np.asarray(ts_dict["activity"], dtype=float)
    times_dt = mdates.date2num([datetime.utcfromtimestamp(val) for val in t])
    if times_dt.size < 2:
        coeff = np.array([0.0, a[0] if a.size else 0.0])
    else:
        coeff = np.polyfit(times_dt, a, 1)
    fig, ax = plt.subplots()
    ax.plot(times_dt, a, "o")
    ax.plot(
        times_dt,
        np.polyval(coeff, times_dt),
        label=f"slope={coeff[0] / 86400.0:.2e} Bq/s",
    )
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.legend()
    _format_time_axes(ax, times_dt)
    fig.autofmt_xdate()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_trend")

