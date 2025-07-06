"""Baseline subtraction utilities for radon analysis."""

from __future__ import annotations

import numpy as np

__all__ = ["subtract_baseline_counts", "subtract_baseline_rate"]


def _validate_subtraction(isotopes):
    allowed = {"noise"}
    if not set(isotopes) <= allowed:
        raise ValueError("Only 'noise' may be subtracted from radon baseline")


def _scaling_factor(
    dt_window: float,
    dt_baseline: float,
    err_window: float = 0.0,
    err_baseline: float = 0.0,
) -> tuple[float, float]:
    """Return scaling factor between analysis and baseline durations."""

    if dt_baseline == 0:
        raise ValueError("dt_baseline must be non-zero")

    scale = float(dt_window) / float(dt_baseline)
    var = (err_window / dt_baseline) ** 2
    var += ((dt_window * err_baseline) / dt_baseline**2) ** 2
    return scale, float(np.sqrt(var))


def subtract_baseline_counts(
    counts: float,
    efficiency: float,
    live_time: float,
    baseline_counts: float,
    baseline_live_time: float,
    isotopes=None,
) -> tuple[float, float]:
    """Return background-corrected rate and uncertainty."""

    if isotopes is not None:
        _validate_subtraction(isotopes)

    if live_time <= 0:
        raise ValueError("live_time must be positive for baseline correction")
    if baseline_live_time <= 0:
        raise ValueError("baseline_live_time must be positive for baseline correction")
    if efficiency <= 0:
        raise ValueError("efficiency must be positive for baseline correction")

    scale, _ = _scaling_factor(live_time, baseline_live_time)

    net = counts - scale * baseline_counts
    corrected_rate = net / live_time / efficiency

    sigma_sq = counts / live_time**2 / efficiency**2
    baseline_sigma_sq = baseline_counts * scale**2 / live_time**2 / efficiency**2
    corrected_sigma = np.sqrt(sigma_sq + baseline_sigma_sq)
    return corrected_rate, corrected_sigma


def subtract_baseline_rate(
    fit_rate: float,
    fit_sigma: float,
    counts: float,
    efficiency: float,
    live_time: float,
    baseline_counts: float,
    baseline_live_time: float,
    scale: float = 1.0,
) -> tuple[float, float, float, float]:
    """Apply baseline subtraction to a fitted decay rate."""

    if live_time <= 0:
        raise ValueError("live_time must be positive for baseline correction")
    if baseline_live_time <= 0:
        raise ValueError("baseline_live_time must be positive for baseline correction")
    if efficiency <= 0:
        raise ValueError("efficiency must be positive for baseline correction")

    baseline_rate = baseline_counts / (baseline_live_time * efficiency)
    baseline_sigma = np.sqrt(baseline_counts) / (baseline_live_time * efficiency)

    _, sigma_rate = subtract_baseline_counts(
        counts,
        efficiency,
        live_time,
        baseline_counts,
        baseline_live_time,
    )

    corrected_rate = fit_rate - scale * baseline_rate
    corrected_sigma = float(np.hypot(fit_sigma, sigma_rate * scale))

    return corrected_rate, corrected_sigma, baseline_rate, baseline_sigma
