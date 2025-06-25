import numpy as np
import logging
import pandas as pd
from utils import parse_datetime

from baseline_utils import subtract_baseline_dataframe

__all__ = ["rate_histogram", "subtract_baseline", "subtract_baseline_dataframe"]


def _to_datetime64(events: pd.DataFrame) -> np.ndarray:
    """Return numpy.ndarray[datetime64[ns, UTC]]."""

    ts_col = events["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts_col):
        ser = ts_col
    else:
        ser = ts_col.map(parse_datetime)

    if getattr(ser.dtype, "tz", None) is not None:
        ser = ser.dt.tz_convert("UTC")

    ts = ser.view("int64").to_numpy()
    return np.asarray(ts)


def rate_histogram(df, bins):
    """Return (histogram in counts/s, live_time_s)."""
    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _to_datetime64(df)
    live = float(ts[-1] - ts[0]) / 1e9
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
