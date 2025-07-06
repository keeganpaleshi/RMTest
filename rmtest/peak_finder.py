"""Simple utilities to plot isotope peaks in a consistent legend order."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt


def plot_peak(ax: plt.Axes, x: Sequence[float], y: Sequence[float], *, label: str, **kwargs) -> None:
    """Plot a single peak on ``ax``."""
    ax.plot(x, y, label=label, **kwargs)


def plot_peaks(ax: plt.Axes, peaks: Mapping[str, Iterable[Sequence[float]]], **kwargs) -> None:
    """Plot multiple peaks in a fixed legend order.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    peaks : mapping
        Dictionary mapping isotope name to ``(x, y)`` data.
    """
    for iso in ("Po210", "Po218", "Po214"):
        data = peaks.get(iso)
        if data is None:
            continue
        x, y = data
        plot_peak(ax, x, y, label=iso, **kwargs)

    ax.legend()

