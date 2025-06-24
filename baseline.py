import numpy as np
import logging
import pandas as pd
from utils import parse_timestamp

__all__ = ["rate_histogram", "subtract_baseline"]


def _scaling_factor(dt_window: float, dt_baseline: float,
                    err_window: float = 0.0,
                    err_baseline: float = 0.0) -> tuple[float, float]:
    """Return scaling factor between analysis and baseline durations.

    This helper computes ``dt_window / dt_baseline`` and propagates the
    1-sigma uncertainty from ``err_window`` and ``err_baseline`` assuming they
    are independent.  A ``ValueError`` is raised when ``dt_baseline`` is zero
    to avoid division by zero.

    Parameters
    ----------
    dt_window : float
        Duration of the analysis window in seconds.
    dt_baseline : float
        Duration of the baseline interval in seconds.
    err_window : float, optional
        Uncertainty on ``dt_window``. Default is ``0.0``.
    err_baseline : float, optional
        Uncertainty on ``dt_baseline``. Default is ``0.0``.

    Returns
    -------
    float
        The scaling factor ``dt_window / dt_baseline``.
    float
        Propagated uncertainty on the scaling factor.

    Examples
    --------
    >>> _scaling_factor(10.0, 5.0)
    (2.0, 0.0)
    >>> _scaling_factor(10.0, 5.0, 0.1, 0.2)
    (2.0, 0.0894427191)
    """

    if dt_baseline == 0:
        raise ValueError("dt_baseline must be non-zero")

    scale = float(dt_window) / float(dt_baseline)
    var = (err_window / dt_baseline) ** 2
    var += ((dt_window * err_baseline) / dt_baseline**2) ** 2
    return scale, float(np.sqrt(var))


def _seconds(col):
    """Return timestamp column as seconds from epoch."""
    ts = col
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts.view("int64").to_numpy() / 1e9
    ts = ts.map(parse_timestamp)
    return ts.astype(float).to_numpy()


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
    t0 = parse_timestamp(t_base0)
    t1 = parse_timestamp(t_base1)
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
