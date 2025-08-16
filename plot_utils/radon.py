import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    t = np.asarray(ts_dict["time"])
    a = np.asarray(ts_dict["activity"]) * 1e3
    e = np.asarray(ts_dict["error"]) * 1e3
    fig, ax = plt.subplots()
    ax.errorbar(t, a, yerr=e, fmt="o")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.set_ylabel("Rn-222 activity [mBq]")
    ax.set_xlabel("Time (UTC)")
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_activity")


def plot_radon_trend(ts_dict, outdir: Path, out_png: str | Path | None = None) -> None:
    t = np.asarray(ts_dict["time"])
    a = np.asarray(ts_dict["activity"]) * 1e3
    if t.size < 2:
        coeff = np.array([0.0, a[0] if a.size else 0.0])
    else:
        coeff = np.polyfit(t, a, 1)
    fig, ax = plt.subplots()
    ax.plot(t, a, "o")
    ax.plot(t, np.polyval(coeff, t), label=f"slope={coeff[0]:.2e} mBq/s")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.set_ylabel("Rn-222 activity [mBq]")
    ax.set_xlabel("Time (UTC)")
    ax.legend()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        _save(fig, outdir, "radon_trend")
