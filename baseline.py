import numpy as np
import logging
import pandas as pd
from utils import parse_datetime

from baseline_utils import subtract_baseline_dataframe

__all__ = [
    "compute_rate_histogram",
    "subtract_baseline_df",
    "subtract_baseline_dataframe",
    "subtract_baseline",
]


def _to_datetime64(col):
    """Return timestamp column as ``numpy.datetime64`` values in UTC."""

    if pd.api.types.is_datetime64_any_dtype(col):
        ser = col
        if getattr(ser.dtype, "tz", None) is not None:
            ser = ser.dt.tz_convert("UTC").dt.tz_localize(None)
        ts = ser.astype("datetime64[ns]").to_numpy()
    else:
        ts = col.map(parse_datetime).astype("datetime64[ns]").to_numpy()
    return np.asarray(ts)


def compute_rate_histogram(df, bins):
    """Return (histogram in counts/s, live_time_s)."""
    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _to_datetime64(df["timestamp"])
    live = float((ts[-1] - ts[0]) / np.timedelta64(1, "s"))
    hist_src = df.get("subtracted_adc_hist", df["adc"]).to_numpy()
    hist, _ = np.histogram(hist_src, bins=bins)
    if live <= 0:
        return np.zeros_like(hist, dtype=float), live
    return hist / live, live


def subtract_baseline_df(df_analysis, df_full, bins, t_base0, t_base1,
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


# Backwards compatibility alias
subtract_baseline = subtract_baseline_df

