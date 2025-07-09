import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir: Path) -> None:
    t = np.asarray(ts_dict["time"])
    a = np.asarray(ts_dict["activity"])
    e = np.asarray(ts_dict["error"])
    fig, ax = plt.subplots()
    ax.errorbar(t, a, yerr=e, fmt="o")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    _save(fig, outdir, "radon_activity")


def plot_radon_trend(ts_dict, outdir: Path) -> None:
    t = np.asarray(ts_dict["time"])
    a = np.asarray(ts_dict["activity"])
    if t.size < 2:
        coeff = np.array([0.0, a[0] if a.size else 0.0])
    else:
        coeff = np.polyfit(t, a, 1)
    fig, ax = plt.subplots()
    ax.plot(t, a, "o")
    ax.plot(t, np.polyval(coeff, t), label=f"slope={coeff[0]:.2e} Bq/s")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.legend()
    _save(fig, outdir, "radon_trend")
