import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
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
    ax.errorbar(ts.time, ts.activity, yerr=getattr(ts, "error", None), fmt="o")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    fig.tight_layout()
    fig.savefig(outdir / "radon_activity.png", dpi=300)
    plt.close(fig)


def plot_radon_trend(ts, outdir):
    """Plot radon activity trend without uncertainties."""
    outdir = Path(outdir)
    fig, ax = plt.subplots()
    ax.plot(ts.time, ts.activity, "o-")
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    fig.tight_layout()
    fig.savefig(outdir / "radon_trend.png", dpi=300)
    plt.close(fig)
