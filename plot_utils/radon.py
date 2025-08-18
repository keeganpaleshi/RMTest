import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ._time_utils import guard_time_alias, setup_time_axis, to_mpl_times


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    guard_time_alias(locals())
    times = to_mpl_times(ts_dict["time"])
    activity = np.asarray(ts_dict["activity"], dtype=float)
    errors = np.asarray(ts_dict["error"], dtype=float)
    fig, ax = plt.subplots()
    ax.errorbar(times, activity, yerr=errors, fmt="o")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_activity")


def plot_radon_trend(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    guard_time_alias(locals())
    times = to_mpl_times(ts_dict["time"])
    activity = np.asarray(ts_dict["activity"], dtype=float)
    if times.size < 2:
        coeff = np.array([0.0, activity[0] if activity.size else 0.0])
    else:
        coeff = np.polyfit(times, activity, 1)
    fig, ax = plt.subplots()
    ax.plot(times, activity, "o")
    ax.plot(times, np.polyval(coeff, times), label=f"slope={coeff[0]:.2e} Bq/s")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend()
    setup_time_axis(ax, times)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_trend")
