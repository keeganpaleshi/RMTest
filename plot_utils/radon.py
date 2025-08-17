import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    t = np.asarray(ts_dict["time"], dtype=float)
    a = np.asarray(ts_dict["activity"], dtype=float)
    e = np.asarray(ts_dict["error"], dtype=float)
    times_dt = mdates.epoch2num(t)
    fig, ax = plt.subplots()
    ax.errorbar(times_dt, a, yerr=e, fmt="o")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")

    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.get_offset_text().set_visible(False)
    t0 = times_dt[0] if times_dt.size else 0.0
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: (x - t0) * 24.0, lambda h: h / 24.0 + t0),
    )
    secax.set_xlabel("Elapsed time (h)")
    secax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    secax.xaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_activity")


def plot_radon_trend(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    t = np.asarray(ts_dict["time"], dtype=float)
    a = np.asarray(ts_dict["activity"], dtype=float)
    times_dt = mdates.epoch2num(t)
    if t.size < 2:
        coeff = np.array([0.0, a[0] if a.size else 0.0])
    else:
        coeff = np.polyfit(t, a, 1)
    fig, ax = plt.subplots()
    ax.plot(times_dt, a, "o")
    ax.plot(times_dt, np.polyval(coeff, t), label=f"slope={coeff[0]:.2e} Bq/s")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.legend()

    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.get_offset_text().set_visible(False)
    t0 = times_dt[0] if times_dt.size else 0.0
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: (x - t0) * 24.0, lambda h: h / 24.0 + t0),
    )
    secax.set_xlabel("Elapsed time (h)")
    secax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    secax.xaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_trend")
