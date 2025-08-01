import logging
from datetime import datetime

import numpy as np
import pandas as pd

from utils.time_utils import parse_timestamp, tz_localize_utc, tz_convert_utc
from io_utils import load_events
from radon.baseline import (
    subtract_baseline_counts,
    subtract_baseline_rate,
)


class BaselineError(RuntimeError):
    """Raised when baseline subtraction diagnostics fail."""

    pass


__all__ = [
    "BaselineError",
    "compute_dilution_factor",
    "apply_baseline_subtraction",
    "subtract_baseline_dataframe",
    "subtract_baseline_counts",
    "subtract_baseline_rate",
    "rate_histogram",
    "subtract",
    "summarize_baseline",
    "baseline_period_before_data",
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


def baseline_period_before_data(baseline_end, data_start):
    """Return ``True`` if the baseline interval ends before data starts.

    Parameters
    ----------
    baseline_end, data_start:
        ``str``, numeric seconds or ``datetime`` objects.  Values may be
        timezone-naive or timezone-aware and are interpreted in UTC.

    Notes
    -----
    Inputs are converted to UTC using :func:`utils.time_utils.parse_timestamp`
    and compared on their integer nanoseconds to avoid issues with mixed time
    zone information.
    """

    end_ns = parse_timestamp(pd.Timestamp(baseline_end)).value
    start_ns = parse_timestamp(pd.Timestamp(data_start)).value
    return end_ns < start_ns


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
            ser = tz_localize_utc(ser)
    else:
        ser = col.map(parse_timestamp)

    ser = tz_convert_utc(ser)
    return ser.to_numpy(dtype="datetime64[ns]")


def rate_histogram(df: pd.DataFrame, bins) -> tuple[np.ndarray, float]:
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
    *,
    allow_fallback: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return baseline-corrected ``DataFrame`` and histogram.

    ``t_base0`` and ``t_base1`` may be naïve or timezone-aware. They are
    interpreted in UTC and compared using integer nanoseconds to avoid
    issues with differing time zone information between ``df_full`` and
    the provided range. When ``baseline_range`` matches no events a warning is
    logged and a copy of ``df_analysis`` is returned.
    """

    assert mode in ("none", "electronics", "radon", "all")

    if mode == "none":
        rate_an, live_an = rate_histogram(df_analysis, bins)
        if live_time_analysis is None:
            live_time_analysis = live_an
        hist = rate_an * live_time_analysis
        return df_analysis.copy(), hist

    rate_an, live_an = rate_histogram(df_analysis, bins)
    if live_time_analysis is None:
        live_time_analysis = live_an

    t0 = parse_timestamp(t_base0)
    t1 = parse_timestamp(t_base1)
    ts_full = _to_datetime64(df_full)
    if ts_full.size > 0:
        events_start = pd.Timestamp(ts_full.min(), tz="UTC")
        events_end = pd.Timestamp(ts_full.max(), tz="UTC")
        if t1 < events_start or t0 > events_end:
            logging.warning(
                "Baseline interval outside data range – taking counts anyway"
            )
    ts_int = ts_full.view("int64")
    t0_ns = t0.value
    t1_ns = t1.value
    mask = (ts_int >= t0_ns) & (ts_int <= t1_ns)
    if not mask.any():
        logging.warning(
            "baseline_range matched no events – subtraction skipped"
        )
        hist = rate_an * live_time_analysis
        return df_analysis.copy(), hist

    baseline_subset = df_full.loc[mask]

    rate_bl, live_bl = rate_histogram(baseline_subset, bins)

    if mode in ("electronics", "radon", "all"):
        net_counts = (rate_an - rate_bl) * live_time_analysis
    else:  # mode == "none"
        net_counts = rate_an * live_time_analysis

    df_out = df_analysis.copy()
    return df_out, net_counts


def subtract(
    df_analysis: pd.DataFrame,
    df_full: pd.DataFrame,
    bins,
    t_base0: datetime,
    t_base1: datetime,
    mode: str = "all",
    *,
    live_time_analysis: float | None = None,
    allow_fallback: bool = False,
    **kw,
) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray]]:
    """Return corrected ``DataFrame`` and (histogram, error).

    Uncertainties are propagated in quadrature from the analysis and
    baseline histograms unless ``kw.get("uncert_prop") == "none"``.  When the
    baseline interval contains no events a warning is logged and
    ``df_analysis`` is returned unchanged.
    """

    df_corr, hist = apply_baseline_subtraction(
        df_analysis,
        df_full,
        bins,
        t_base0,
        t_base1,
        mode=mode,
        live_time_analysis=live_time_analysis,
        allow_fallback=allow_fallback,
    )

    if kw.get("uncert_prop") == "none":
        err_hist = np.zeros(len(bins) - 1, dtype=float)
    else:
        rate_an, live_an = rate_histogram(df_analysis, bins)
        if live_time_analysis is None:
            live_time_analysis = live_an

        t0 = parse_timestamp(t_base0)
        t1 = parse_timestamp(t_base1)
        ts_full = _to_datetime64(df_full)
        ts_int = ts_full.view("int64")
        mask = (ts_int >= t0.value) & (ts_int <= t1.value)

        if mask.any():
            rate_bl, live_bl = rate_histogram(df_full.loc[mask], bins)
        else:
            rate_bl, live_bl = np.zeros(len(bins) - 1, dtype=float), 0.0

        counts_an = rate_an * live_time_analysis
        counts_bl = rate_bl * live_bl
        scale = live_time_analysis / live_bl if live_bl > 0 else 0.0

        var = counts_an + (scale**2) * counts_bl
        err_hist = np.sqrt(var)

    return df_corr, (hist, err_hist)


# Backwards compatibility
subtract_baseline_dataframe = apply_baseline_subtraction


# thin wrappers are imported above from :mod:`radon.baseline`


def summarize_baseline(
    cfg: dict, isotopes: list[str]
) -> dict[str, tuple[float, float, float]]:
    """Return baseline subtraction summary for selected isotopes.

    Parameters
    ----------
    cfg : dict
        Summary dictionary containing ``baseline`` and ``time_fit`` sections.
    isotopes : list[str]
        Isotope names to include in the output.

    Returns
    -------
    dict
        Mapping ``iso -> (raw_rate, baseline_rate, corrected_rate)`` in Bq.

    Raises
    ------
    BaselineError
        If any corrected rate is negative and ``cfg.get('allow_negative_baseline')``
        is not ``True``.
    """

    baseline = cfg.get("baseline", {})
    tf = cfg.get("time_fit", {})
    scales = baseline.get("scales", {})
    base_rates = baseline.get("rate_Bq", {})

    allow_negative = bool(cfg.get("allow_negative_baseline"))

    summary: dict[str, tuple[float, float, float]] = {}
    for iso in isotopes:
        fit = tf.get(iso, {})
        raw = fit.get(f"E_{iso}")
        if raw is None:
            continue
        base = float(base_rates.get(iso, 0.0)) * float(scales.get(iso, 1.0))
        corr = float(fit.get("E_corrected", raw - base))
        raw = float(raw)
        summary[iso] = (raw, base, corr)
        if corr < 0 and not allow_negative:
            raise BaselineError(f"negative corrected rate for {iso}")

    return summary
