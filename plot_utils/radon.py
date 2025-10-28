import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ._time_utils import guard_mpl_times, setup_time_axis


def _autoscale_with_errors(ax, values, errors):
    """Expand ``ax`` limits to include error bars with a small margin."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return

    finite = np.isfinite(arr)
    if not finite.any():
        return

    lower = arr[finite].min()
    upper = arr[finite].max()

    if errors is not None:
        err_arr = np.asarray(errors, dtype=float)
        if err_arr.size == arr.size:
            lower_vals = arr - err_arr
            upper_vals = arr + err_arr
            finite_lower = np.isfinite(lower_vals)
            if finite_lower.any():
                lower = min(lower, lower_vals[finite_lower].min())
            finite_upper = np.isfinite(upper_vals)
            if finite_upper.any():
                upper = max(upper, upper_vals[finite_upper].max())

    if not np.isfinite(lower) or not np.isfinite(upper):
        return

    if lower == upper:
        scale = abs(lower) if lower != 0 else 1.0
        margin = 0.05 * scale
        lower -= margin
        upper += margin
    else:
        margin = 0.05 * (upper - lower)
        lower -= margin
        upper += margin

    ax.set_ylim(lower, upper)


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    times_mpl = guard_mpl_times(times=ts_dict["time"])
    activity = np.asarray(ts_dict["activity"], dtype=float)
    errors_raw = ts_dict.get("error")
    errors = None if errors_raw is None else np.asarray(errors_raw, dtype=float)
    fig, ax = plt.subplots()
    ax.errorbar(times_mpl, activity, yerr=errors, fmt="o")
    ax.set_ylabel("Rn-222 concentration [Bq/L]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    _autoscale_with_errors(ax, activity, errors)
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


def plot_radon_activity_elapsed(
    times,
    activity,
    outdir: Path,
    out_png: str | Path | None = None,
    *,
    errors=None,
) -> None:
    """Plot radon concentration versus elapsed hours."""

    arr_times = np.asarray(list(times), dtype=float)
    activity_arr = np.asarray(activity, dtype=float)
    if arr_times.size == 0 or activity_arr.size == 0:
        return

    errors_arr = None if errors is None else np.asarray(errors, dtype=float)
    hours = (arr_times - arr_times[0]) / 3600.0

    fig, ax = plt.subplots()
    ax.errorbar(hours, activity_arr, yerr=errors_arr, fmt="o")
    ax.set_xlabel("Elapsed Time (h)")
    ax.set_ylabel("Rn-222 concentration [Bq/L]")
    ax.ticklabel_format(axis="y", style="plain")
    _autoscale_with_errors(ax, activity_arr, errors_arr)
    ax.yaxis.get_offset_text().set_visible(False)
    fig.tight_layout()

    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_activity_elapsed")
