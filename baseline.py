import numpy as np
import logging
import pandas as pd

from radon import baseline as _baseline
import baseline_utils

subtract_baseline = baseline_utils.apply_baseline_correction

__all__ = ["rate_histogram", "subtract_baseline", "apply_baseline_correction"]


def rate_histogram(df, bins):
    """Return ``(histogram, live_time_s)`` for a timestamped ``DataFrame``.

    The timestamp column may be timezone-aware.  Internally timestamps are
    converted to UTC and differences are computed using the underlying
    integer nanoseconds to avoid dtype mismatches.
    """
    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _baseline._to_datetime64(df["timestamp"])
    ts_int = ts.view("int64")
    live = float((ts_int[-1] - ts_int[0]) / 1e9)
    hist_src = df.get("subtracted_adc_hist", df["adc"]).to_numpy()
    hist, _ = np.histogram(hist_src, bins=bins)
    if live <= 0:
        return np.zeros_like(hist, dtype=float), live
    return hist / live, live
