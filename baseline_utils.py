import logging
from datetime import datetime

import numpy as np
import pandas as pd

from utils import parse_datetime
from radon.baseline import (
    subtract_baseline_counts,
    subtract_baseline_rate,
    _scaling_factor,
)

__all__ = [
    "compute_dilution_factor",
    "subtract_baseline_dataframe",
    "subtract_baseline_counts",
    "subtract_baseline_rate",
    "_scaling_factor",
]




def compute_dilution_factor(monitor_volume: float, sample_volume: float) -> float:
    """Return dilution factor ``monitor_volume / (monitor_volume + sample_volume)``.

    Returns zero when the combined volume is non-positive to avoid division by
    zero or negative scales.
    """

    total = monitor_volume + sample_volume
    if total <= 0:
        return 0.0
    return float(monitor_volume) / float(total)


def _to_datetime64(events: pd.DataFrame | pd.Series) -> np.ndarray:
    """Return ``numpy.ndarray`` of ``datetime64[ns]`` in UTC.

    Parameters
    ----------
    events:
        ``Series`` of timestamps or a ``DataFrame`` containing a
        ``"timestamp"`` column.

    Both timezone-naive and timezone-aware inputs are supported.  Any
    timezone information is converted to UTC before the underlying
    integer nanoseconds are reinterpreted as ``datetime64[ns]``.  This
    avoids ``pandas`` warnings when comparing arrays with different time
    zone attributes.
    """

    if isinstance(events, pd.DataFrame):
        col = events["timestamp"]
    else:
        col = events

    if pd.api.types.is_datetime64_any_dtype(col):
        ser = col
        if getattr(ser.dtype, "tz", None) is None:
            ser = ser.dt.tz_localize("UTC")
    else:
        ser = col.map(parse_datetime)

    ser = ser.dt.tz_convert("UTC")
    return ser.to_numpy(dtype="datetime64[ns]")


def _rate_histogram(df: pd.DataFrame, bins) -> tuple[np.ndarray, float]:
    """Return histogram in counts/s and the live time in seconds.

    Timestamp columns may be timezone-aware.  Differences are computed
    on the underlying integer nanoseconds to avoid issues with mixed
    time zones.
    """

    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    ts = _to_datetime64(df)
    ts_int = ts.view("int64")
    live = float((ts_int[-1] - ts_int[0]) / 1e9)
    hist_src = df.get("subtracted_adc_hist", df["adc"]).to_numpy()
    hist, _ = np.histogram(hist_src, bins=bins)
    if live <= 0:
        return np.zeros_like(hist, dtype=float), live
    return hist / live, live


def apply_baseline_subtraction(
    df_analysis: pd.DataFrame,
    df_full: pd.DataFrame,
    bins,
    t_base0: datetime,
    t_base1: datetime,
    mode: str = "all",
    live_time_analysis: float | None = None,
) -> pd.DataFrame:
    """Return new ``DataFrame`` with baseline-subtracted spectra.

    ``t_base0`` and ``t_base1`` may be naïve or timezone-aware.  They are
    interpreted in UTC and compared using integer nanoseconds to avoid
    issues with differing time zone information between ``df_full`` and
    the provided range.
    """

    assert mode in ("none", "electronics", "radon", "all")

    if mode == "none":
        return df_analysis.copy()

    rate_an, live_an = _rate_histogram(df_analysis, bins)
    if live_time_analysis is None:
        live_time_analysis = live_an

    t0 = parse_datetime(t_base0)
    t1 = parse_datetime(t_base1)
    ts_full = _to_datetime64(df_full)
    ts_int = ts_full.view("int64")
    t0_ns = t0.value
    t1_ns = t1.value
    mask = (ts_int >= t0_ns) & (ts_int <= t1_ns)
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


# Backwards compatibility
subtract_baseline_dataframe = apply_baseline_subtraction



# thin wrappers are imported above from :mod:`radon.baseline`
