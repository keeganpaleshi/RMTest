import numpy as np
import logging
import pandas as pd
from utils import parse_time
from io_utils import parse_datetime

__all__ = ["rate_histogram", "subtract_baseline", "_scaling_factor"]


def _seconds(col):
    """Return timestamp column as seconds from epoch."""
    ts = col
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = ts.map(parse_datetime)
    ts = pd.to_datetime(ts, utc=True)
    return ts.view("int64").to_numpy() / 1e9


def rate_histogram(df, bins):
    """Return (histogram in counts/s, live_time_s)."""
    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _seconds(df["timestamp"])
    live = float(ts[-1] - ts[0])
    hist_src = df.get("subtracted_adc_hist", df["adc"]).to_numpy()
    hist, _ = np.histogram(hist_src, bins=bins)
    if live <= 0:
        return np.zeros_like(hist, dtype=float), live
    return hist / live, live


def _scaling_factor(baseline_start, baseline_stop, window_start, window_stop):
    """Return the factor mapping baseline counts to the analysis window.

    The returned scale converts counts from a baseline interval into the
    time-normalised space of the analysis window:

        scale = live_time_window / live_time_baseline

    so that::

        N_scaled = scale * N_baseline

    Propagated uncertainty follows::

        sigma^2 = N_sig + scale**2 * N_base
    """

    dt_baseline = (baseline_stop - baseline_start).total_seconds()
    dt_window = (window_stop - window_start).total_seconds()
    if dt_baseline == 0:
        raise ValueError("Baseline duration is zero")
    return dt_window / dt_baseline


def subtract_baseline(df_analysis, df_full, bins, t_base0, t_base1,
                      mode="all", live_time_analysis=None):
    """Subtract baseline from ``df_analysis`` and return a new ``DataFrame``.

    Parameters
    ----------
    df_analysis : pandas.DataFrame
        Data to be baseline corrected.
    df_full : pandas.DataFrame
        Full dataset used to extract the baseline slice.
    bins : array-like
        Histogram bin edges.
    t_base0, t_base1 : datetime.datetime
        Start and end of the baseline range (UTC assumed for naive datetimes).
    mode : {"none", "electronics", "radon", "all"}
        Type of subtraction to perform.
    live_time_analysis : float, optional
        Seconds represented by ``df_analysis``; if ``None`` it is calculated
        internally.
    """
    assert mode in ("none", "electronics", "radon", "all")

    if mode == "none":
        return df_analysis.copy()

    # analysis spectrum (counts/s)
    rate_an, live_an = rate_histogram(df_analysis, bins)
    if live_time_analysis is None:
        live_time_analysis = live_an

    # baseline slice
    t0 = parse_time(t_base0)
    t1 = parse_time(t_base1)
    ts_full = _seconds(df_full["timestamp"])
    mask = (ts_full >= t0) & (ts_full <= t1)
    if not mask.any():
        logging.warning("baseline_range matched no events – skipping subtraction")
        return df_analysis.copy()

    rate_bl, live_bl = rate_histogram(df_full.loc[mask], bins)

    # currently electronics vs radon use same subtraction; future hooks can differ
    if mode in ("electronics", "radon", "all"):
        net_counts = (rate_an - rate_bl) * live_time_analysis
    else:  # mode == "none"
        net_counts = rate_an * live_time_analysis

    df_out = df_analysis.copy()
    df_out["subtracted_adc_hist"] = [net_counts] * len(df_out)
    return df_out
