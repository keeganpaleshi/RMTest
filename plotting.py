import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

__all__ = ["plot_radon_activity", "plot_radon_trend"]


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
    activity = np.asarray(ts.activity, dtype=float) * 1e3
    errs = getattr(ts, "error", None)
    if errs is not None:
        errs = np.asarray(errs, dtype=float) * 1e3
    ax.errorbar(ts.time, activity, yerr=errs, fmt="o")
    ax.set_ylabel("Radon activity [mBq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    fig.tight_layout()
    fig.savefig(outdir / "radon_activity.png", dpi=300)
    plt.close(fig)


def plot_radon_trend(ts, outdir):
    """Plot radon activity trend without uncertainties."""
    outdir = Path(outdir)
    fig, ax = plt.subplots()
    activity = np.asarray(ts.activity, dtype=float) * 1e3
    ax.plot(ts.time, activity, "o-")
    ax.set_ylabel("Radon activity [mBq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    fig.tight_layout()
    fig.savefig(outdir / "radon_trend.png", dpi=300)
    plt.close(fig)
