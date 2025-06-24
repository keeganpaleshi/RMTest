import logging
import numpy as np
import pandas as pd

from utils import parse_datetime

__all__ = [
    "scaling_factor",
    "dilution_factor",
    "rate_histogram",
    "subtract_baseline_df",
    "subtract_baseline_counts",
    "subtract_baseline_rate",
]


def scaling_factor(dt_window: float, dt_baseline: float,
                    err_window: float = 0.0,
                    err_baseline: float = 0.0) -> tuple[float, float]:
    """Return scaling factor between analysis and baseline durations."""
    if dt_baseline == 0:
        raise ValueError("dt_baseline must be non-zero")
    scale = float(dt_window) / float(dt_baseline)
    var = (err_window / dt_baseline) ** 2
    var += ((dt_window * err_baseline) / dt_baseline**2) ** 2
    return scale, float(np.sqrt(var))


def dilution_factor(monitor_volume_l: float, sample_volume_l: float) -> float:
    """Return monitor dilution factor for baseline scaling."""
    total = monitor_volume_l + sample_volume_l
    return float(monitor_volume_l) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def _seconds(col):
    """Return timestamp column as ``numpy.datetime64`` values."""
    if pd.api.types.is_datetime64_any_dtype(col):
        ser = col
        if getattr(ser.dtype, "tz", None) is not None:
            ser = ser.dt.tz_convert("UTC").dt.tz_localize(None)
        ts = ser.astype("datetime64[ns]").to_numpy()
    else:
        ts = col.map(parse_datetime).astype("datetime64[ns]").to_numpy()
    return np.asarray(ts)


def rate_histogram(df: pd.DataFrame, bins):
    """Return (histogram in counts/s, live_time_s)."""
    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _seconds(df["timestamp"])
    live = float((ts[-1] - ts[0]) / np.timedelta64(1, "s"))
    hist_src = df.get("subtracted_adc_hist", df["adc"]).to_numpy()
    hist, _ = np.histogram(hist_src, bins=bins)
    if live <= 0:
        return np.zeros_like(hist, dtype=float), live
    return hist / live, live


def subtract_baseline_df(df_analysis: pd.DataFrame,
                          df_full: pd.DataFrame,
                          bins,
                          t_base0,
                          t_base1,
                          mode: str = "all",
                          live_time_analysis: float | None = None) -> pd.DataFrame:
    """Subtract baseline from ``df_analysis`` and return a new ``DataFrame``."""
    assert mode in ("none", "electronics", "radon", "all")

    if mode == "none":
        return df_analysis.copy()

    rate_an, live_an = rate_histogram(df_analysis, bins)
    if live_time_analysis is None:
        live_time_analysis = live_an

    t0 = parse_datetime(t_base0)
    t1 = parse_datetime(t_base1)
    ts_full = _seconds(df_full["timestamp"])
    mask = (ts_full >= t0) & (ts_full <= t1)
    if not mask.any():
        logging.warning("baseline_range matched no events – skipping subtraction")
        return df_analysis.copy()

    rate_bl, _ = rate_histogram(df_full.loc[mask], bins)

    if mode in ("electronics", "radon", "all"):
        net_counts = (rate_an - rate_bl) * live_time_analysis
    else:
        net_counts = rate_an * live_time_analysis

    df_out = df_analysis.copy()
    df_out["subtracted_adc_hist"] = [net_counts] * len(df_out)
    return df_out


# ---------------------------------------------------------------------------
# Scalar baseline subtraction utilities
# ---------------------------------------------------------------------------

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
        raise ValueError(
            "baseline_live_time must be positive for baseline correction"
        )
    if efficiency <= 0:
        raise ValueError("efficiency must be positive for baseline correction")

    scale, _ = scaling_factor(live_time, baseline_live_time)

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
        raise ValueError(
            "baseline_live_time must be positive for baseline correction"
        )
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
