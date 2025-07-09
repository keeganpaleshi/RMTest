import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir: Path, out_png: Path | None = None) -> None:
    """Plot radon activity versus time."""

    t = np.asarray(ts_dict["time"])
    a = np.asarray(ts_dict["activity"])
    e = np.asarray(ts_dict["error"])
    fig, ax = plt.subplots()
    ax.errorbar(t, a, yerr=e, fmt="o")
    ax.set_ylabel("Rn-222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")

    outdir = Path(outdir)
    if out_png is None:
        _save(fig, outdir, "radon_activity")
    else:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)


def plot_radon_trend(ts_dict, outdir: Path, out_png: Path | None = None) -> None:
    """Plot a radon activity trend."""

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

    outdir = Path(outdir)
    if out_png is None:
        _save(fig, outdir, "radon_trend")
    else:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
