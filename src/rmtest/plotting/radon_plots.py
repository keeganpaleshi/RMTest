"""Compatibility wrapper for :mod:`radon_plots`."""

from importlib import import_module


_core = import_module("radon_plots")

plot_rn_inferred_vs_time = _core.plot_rn_inferred_vs_time
plot_ambient_rn_vs_time = _core.plot_ambient_rn_vs_time
plot_volume_equiv_vs_time = _core.plot_volume_equiv_vs_time


__all__ = [
    "plot_rn_inferred_vs_time",
    "plot_ambient_rn_vs_time",
    "plot_volume_equiv_vs_time",
]

