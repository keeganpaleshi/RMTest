import numpy as np
import logging
import pandas as pd

import baseline_utils
from baseline_utils import subtract_baseline_dataframe

__all__ = ["rate_histogram", "subtract_baseline", "subtract_baseline_dataframe"]


def rate_histogram(df, bins):
    """Return ``(histogram in counts/s, live_time_s)``.

    The ``timestamp`` column may be timezone-aware. Values are converted to UTC
    and the live time is computed from integer nanoseconds to ensure consistent
    behaviour irrespective of timezone.
    """
    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = baseline_utils._to_datetime64(df["timestamp"])
    ts_ns = ts.view("int64")
    live = float((ts_ns[-1] - ts_ns[0]) / 1e9)
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
