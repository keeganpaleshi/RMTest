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


def _to_datetime64(col: pd.Series) -> np.ndarray:
    """Return ``datetime64[ns]`` values in UTC.

    ``col`` may contain timezone-aware or naïve timestamps. Any timezone-aware
    values are first converted to UTC before being viewed as ``datetime64[ns]``
    values.  This avoids pandas' implicit timezone conversions when using
    ``astype`` directly.
    """

    if pd.api.types.is_datetime64_any_dtype(col):
        ser = col
        if getattr(ser.dtype, "tz", None) is not None:
            ser = ser.dt.tz_convert("UTC")
        ts = ser.view("int64")
        ts = ts.view("datetime64[ns]")
    else:
        ser = col.map(parse_datetime)
        if getattr(ser.dtype, "tz", None) is not None:
            ser = ser.dt.tz_convert("UTC")
        ts = ser.view("int64")
        ts = ts.view("datetime64[ns]")
    return np.asarray(ts)


def _rate_histogram(df: pd.DataFrame, bins) -> tuple[np.ndarray, float]:
    """Return histogram in counts/s and the live time in seconds.

    The ``timestamp`` column may contain timezone-aware values. They are
    converted to UTC before computing the live time using integer nanoseconds
    to avoid any timezone-related rounding issues.
    """

    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _to_datetime64(df["timestamp"])
    ts_ns = ts.view("int64")
    live = float((ts_ns[-1] - ts_ns[0]) / 1e9)
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
    """Return new ``DataFrame`` with baseline-subtracted spectra.

    ``t_base0`` and ``t_base1`` may be timezone-aware. They are converted to
    integer nanoseconds for comparison with the ``timestamp`` column to avoid
    ambiguities due to mixed timezone information.
    """

    assert mode in ("none", "electronics", "radon", "all")

    if mode == "none":
        return df_analysis.copy()

    rate_an, live_an = _rate_histogram(df_analysis, bins)
    if live_time_analysis is None:
        live_time_analysis = live_an

    t0_ns = parse_datetime(t_base0).to_datetime64().view("int64")
    t1_ns = parse_datetime(t_base1).to_datetime64().view("int64")
    ts_full = _to_datetime64(df_full["timestamp"])
    ts_full_ns = ts_full.view("int64")
    mask = (ts_full_ns >= t0_ns) & (ts_full_ns <= t1_ns)
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
