import numpy as np
import logging
import pandas as pd

from baseline_utils import subtract_baseline_dataframe

__all__ = ["rate_histogram", "subtract_baseline", "subtract_baseline_dataframe"]


def _to_datetime64(col):
    """Return timestamp column as ``datetime64[ns, UTC]`` values."""

    return pd.to_datetime(col, utc=True)


def rate_histogram(df, bins):
    """Return (histogram in counts/s, live_time_s)."""
    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _to_datetime64(df["timestamp"])
    t_start = ts.iloc[0] if hasattr(ts, "iloc") else ts[0]
    t_end = ts.iloc[-1] if hasattr(ts, "iloc") else ts[-1]
    live = float((t_end - t_start).total_seconds())
    hist_src = df.get("subtracted_adc_hist", df["adc"]).to_numpy()
    hist, _ = np.histogram(hist_src, bins=bins)
    if live <= 0:
        return np.zeros_like(hist, dtype=float), live
    return hist / live, live


def subtract_baseline(df_analysis, df_full, bins, t_base0, t_base1,
                      mode="all", live_time_analysis=None):
    """Wrapper for :func:`baseline_utils.subtract_baseline_dataframe`."""

    return subtract_baseline_dataframe(
        df_analysis,
        df_full,
        bins,
        t_base0,
        t_base1,
        mode=mode,
        live_time_analysis=live_time_analysis,
    )
