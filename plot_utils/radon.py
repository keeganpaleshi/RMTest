import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ._time_utils import guard_mpl_times, setup_time_axis


_ERRORBAR_STYLE = {
    "fmt": "o",
    "capsize": 3,
    "capthick": 1,
    "elinewidth": 1,
    "barsabove": True,
}


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    times_mpl = guard_mpl_times(times=ts_dict["time"])
    activity = np.asarray(ts_dict["activity"], dtype=float)
    errors = np.asarray(ts_dict["error"], dtype=float)
    fig, ax = plt.subplots()
    ax.errorbar(times_mpl, activity, yerr=errors, **_ERRORBAR_STYLE)
    ax.set_ylabel("Rn-222 concentration [Bq/L]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_activity")


def plot_radon_trend(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    times_mpl = guard_mpl_times(times=ts_dict["time"])
    activity = np.asarray(ts_dict["activity"], dtype=float)
    if times_mpl.size < 2:
        coeff = np.array([0.0, activity[0] if activity.size else 0.0])
    else:
        coeff = np.polyfit(times_mpl, activity, 1)
    fig, ax = plt.subplots()
    ax.plot(times_mpl, activity, "o")
    ax.plot(
        times_mpl,
        np.polyval(coeff, times_mpl),
        label=f"slope={coeff[0]:.2e} Bq/L/s",
    )
    ax.set_ylabel("Rn-222 concentration [Bq/L]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend()
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_trend")
