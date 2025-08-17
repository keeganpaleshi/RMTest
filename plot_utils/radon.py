import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime
from pathlib import Path


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    t = np.asarray(ts_dict["time"], dtype=float)
    a = np.asarray(ts_dict["activity"], dtype=float)
    e = np.asarray(ts_dict["error"], dtype=float)
    times_dt = mdates.date2num([datetime.utcfromtimestamp(x) for x in t])
    fig, ax = plt.subplots()
    ax.errorbar(times_dt, a, yerr=e, fmt="o")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")

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
    secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos=None: f"{x:g}"))
    secax.set_xlabel("Elapsed Time (h)")

    fig.autofmt_xdate()
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    secax.xaxis.get_offset_text().set_visible(False)
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_activity")


def plot_radon_trend(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    t = np.asarray(ts_dict["time"], dtype=float)
    a = np.asarray(ts_dict["activity"], dtype=float)
    times_dt = mdates.date2num([datetime.utcfromtimestamp(x) for x in t])
    if times_dt.size < 2:
        coeff = np.array([0.0, a[0] if a.size else 0.0])
    else:
        coeff = np.polyfit(times_dt, a, 1)
    fig, ax = plt.subplots()
    ax.plot(times_dt, a, "o")
    ax.plot(times_dt, np.polyval(coeff, times_dt), label=f"slope={coeff[0]:.2e} Bq/s")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend()

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
    secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos=None: f"{x:g}"))
    secax.set_xlabel("Elapsed Time (h)")

    fig.autofmt_xdate()
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    secax.xaxis.get_offset_text().set_visible(False)
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_trend")
