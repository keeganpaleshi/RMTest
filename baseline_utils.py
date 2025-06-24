import logging
from datetime import datetime

import numpy as np
import pandas as pd

from utils import parse_datetime

__all__ = [
    "compute_dilution_factor",
    "subtract_baseline_dataframe",
    "subtract_baseline_counts",
    "subtract_baseline_rate",
    "_scaling_factor",
]


def _scaling_factor(dt_window: float, dt_baseline: float,
                    err_window: float = 0.0,
                    err_baseline: float = 0.0) -> tuple[float, float]:
    """Return scaling factor between analysis and baseline durations.

    This helper computes ``dt_window / dt_baseline`` and propagates the
    1-sigma uncertainty from ``err_window`` and ``err_baseline`` assuming they
    are independent.  A ``ValueError`` is raised when ``dt_baseline`` is zero
    to avoid division by zero.
    """

    if dt_baseline == 0:
        raise ValueError("dt_baseline must be non-zero")

    scale = float(dt_window) / float(dt_baseline)
    var = (err_window / dt_baseline) ** 2
    var += ((dt_window * err_baseline) / dt_baseline**2) ** 2
    return scale, float(np.sqrt(var))


def compute_dilution_factor(monitor_volume: float, sample_volume: float) -> float:
    """Return dilution factor ``monitor_volume / (monitor_volume + sample_volume)``.

    Returns zero when the combined volume is non-positive to avoid division by
    zero or negative scales.
    """

    total = monitor_volume + sample_volume
    if total <= 0:
        return 0.0
    return float(monitor_volume) / float(total)


def _seconds(col: pd.Series) -> np.ndarray:
    """Return timestamp column as ``numpy.datetime64`` values."""

    if pd.api.types.is_datetime64_any_dtype(col):
        ser = col
        if getattr(ser.dtype, "tz", None) is not None:
            ser = ser.dt.tz_convert("UTC").dt.tz_localize(None)
        ts = ser.astype("datetime64[ns]").to_numpy()
    else:
        ts = col.map(parse_datetime).astype("datetime64[ns]").to_numpy()
    return np.asarray(ts)


def _rate_histogram(df: pd.DataFrame, bins) -> tuple[np.ndarray, float]:
    """Return histogram in counts/s and the live time in seconds."""

    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _seconds(df["timestamp"])
    live = float((ts[-1] - ts[0]) / np.timedelta64(1, "s"))
    hist_src = df.get("subtracted_adc_hist", df["adc"]).to_numpy()
    hist, _ = np.histogram(hist_src, bins=bins)
    if live <= 0:
        return np.zeros_like(hist, dtype=float), live
    return hist / live, live


def subtract_baseline_dataframe(
    df_analysis: pd.DataFrame,
    df_full: pd.DataFrame,
    bins,
    t_base0: datetime,
    t_base1: datetime,
    mode: str = "all",
    live_time_analysis: float | None = None,
) -> pd.DataFrame:
    """Return new ``DataFrame`` with baseline-subtracted spectra."""

    assert mode in ("none", "electronics", "radon", "all")

    if mode == "none":
        return df_analysis.copy()

    rate_an, live_an = _rate_histogram(df_analysis, bins)
    if live_time_analysis is None:
        live_time_analysis = live_an

    t0 = parse_datetime(t_base0)
    t1 = parse_datetime(t_base1)
    ts_full = _seconds(df_full["timestamp"])
    mask = (ts_full >= t0) & (ts_full <= t1)
    if not mask.any():
        logging.warning("baseline_range matched no events – skipping subtraction")
        return df_analysis.copy()

    rate_bl, live_bl = _rate_histogram(df_full.loc[mask], bins)

    if mode in ("electronics", "radon", "all"):
        net_counts = (rate_an - rate_bl) * live_time_analysis
    else:  # mode == "none"
        net_counts = rate_an * live_time_analysis

    df_out = df_analysis.copy()
    df_out["subtracted_adc_hist"] = [net_counts] * len(df_out)
    return df_out


def subtract_baseline_counts(
    counts: float,
    efficiency: float,
    live_time: float,
    baseline_counts: float,
    baseline_live_time: float,
) -> tuple[float, float]:
    """Return background-corrected rate and uncertainty."""

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
