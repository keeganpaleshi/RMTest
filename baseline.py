import numpy as np
import logging
import pandas as pd
from utils import parse_datetime

from baseline_utils import _scaling_factor
__all__ = ["rate_histogram", "subtract_baseline"]


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


def rate_histogram(df, bins):
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
    t0 = parse_datetime(t_base0)
    t1 = parse_datetime(t_base1)
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
