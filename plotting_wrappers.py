"""
plotting_wrappers.py

Compatibility wrapper functions for plotting functions.

These wrappers exist to maintain backward compatibility with test code
that expects specific argument signatures or calling conventions.
"""

from pathlib import Path
from plot_utils import (
    plot_radon_activity_full,
    plot_total_radon_full,
    plot_radon_trend_full,
)
from plot_utils.radon import (
    plot_radon_activity as _plot_radon_activity,
    plot_radon_trend as _plot_radon_trend,
)


def plot_radon_activity_dict(ts_dict, outdir, maybe_outdir=None, *_, **__):
    """Compatibility wrapper for tests expecting three arguments with dict input."""
    target = maybe_outdir or outdir
    Path(target).mkdir(parents=True, exist_ok=True)
    return _plot_radon_activity(ts_dict, target)


def plot_radon_trend_dict(ts_dict, outdir, maybe_outdir=None, *_, **__):
    """Compatibility wrapper for tests expecting three arguments with dict input."""
    target = maybe_outdir or outdir
    Path(target).mkdir(parents=True, exist_ok=True)
    return _plot_radon_trend(ts_dict, target)


def plot_radon_activity(
    times,
    activity,
    out_png,
    errors=None,
    *,
    config=None,
    sample_volume_l=None,
    background_mode=None,
):
    """Wrapper used by tests expecting output path as third argument."""

    return plot_radon_activity_full(
        times,
        activity,
        errors,
        out_png,
        config=config,
        sample_volume_l=sample_volume_l,
        background_mode=background_mode,
    )


def plot_total_radon(
    times,
    total_bq,
    out_png,
    errors=None,
    *,
    config=None,
    background_mode=None,
):
    """Wrapper used by tests expecting output path as third argument."""

    return plot_total_radon_full(
        times,
        total_bq,
        errors,
        out_png,
        config=config,
        background_mode=background_mode,
    )


def plot_radon_trend(times, activity, out_png, *, config=None, fit_valid=True):
    """Wrapper used by tests expecting output path as third argument."""
    return plot_radon_trend_full(times, activity, out_png, config=config, fit_valid=fit_valid)
