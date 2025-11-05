import logging
from collections.abc import Mapping
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
    "baseline_allows_negative",
    "baseline_period_before_data",
]


def compute_dilution_factor(monitor_volume: float, sample_volume: float) -> float:
    """Return ``monitor_volume / (monitor_volume + sample_volume)``.

    Parameters
    ----------
    monitor_volume:
        Detector chamber volume in litres. Must be finite and strictly
        positive.
    sample_volume:
        Volume of the sample air introduced to the chamber in litres. Must be
        finite and non-negative.

    Raises
    ------
    ValueError
        If either volume violates the physical constraints above.
    """

    monitor = float(monitor_volume)
    sample = float(sample_volume)
    if not np.isfinite(monitor):
        raise ValueError("monitor_volume must be finite")
    if not np.isfinite(sample):
        raise ValueError("sample_volume must be finite")
    if monitor <= 0:
        raise ValueError("monitor_volume must be positive")
    if sample < 0:
        raise ValueError("sample_volume must be non-negative")

    total = monitor + sample
    if total <= 0:
        # The checks above should prevent this, but guard against unexpected
        # floating-point behaviour.
        raise ValueError("combined volume must be positive")

    return monitor / total


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
    start = int(ts_int.min())
    end = int(ts_int.max())
    live = float((end - start) / 1e9)
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
    logged and a copy of ``df_analysis`` is returned unless ``allow_fallback``
    is ``False``, in which case :class:`BaselineError` is raised.
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
        _min = pd.Timestamp(ts_full.min())
        _max = pd.Timestamp(ts_full.max())
        events_start = _min if _min.tzinfo is not None else _min.tz_localize("UTC")
        events_end = _max if _max.tzinfo is not None else _max.tz_localize("UTC")
        if t1 < events_start or t0 > events_end:
            msg = "Baseline interval outside data range"
            logging.warning("%s – taking counts anyway", msg)
            if not allow_fallback:
                raise BaselineError(msg)
    ts_int = ts_full.view("int64")
    t0_ns = t0.value
    t1_ns = t1.value
    mask = (ts_int >= t0_ns) & (ts_int <= t1_ns)
    if not mask.any():
        msg = "baseline_range matched no events"
        logging.warning("%s – subtraction skipped", msg)
        if not allow_fallback:
            raise BaselineError(msg)
        hist = rate_an * live_time_analysis
        return df_analysis.copy(), hist

    baseline_subset = df_full.loc[mask]

    rate_bl, live_bl = rate_histogram(baseline_subset, bins)

    if mode in ("electronics", "all"):
        net_counts = (rate_an - rate_bl) * live_time_analysis
    else:  # ``mode`` is "none" or "radon"
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

        # ``counts_bl`` may be negative after noise subtraction. The baseline
        # helpers treat them as Poisson variances using their absolute value to
        # avoid unphysical negative uncertainties, so mirror that behaviour
        # here.
        var = counts_an + (scale**2) * np.abs(counts_bl)
        err_hist = np.sqrt(var)

    return df_corr, (hist, err_hist)


# Backwards compatibility
subtract_baseline_dataframe = apply_baseline_subtraction


# thin wrappers are imported above from :mod:`radon.baseline`


def baseline_allows_negative(cfg: dict) -> bool:
    """Return whether the configuration allows negative baseline-corrected values.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing the baseline section.

    Returns
    -------
    bool
        True if negative values are allowed, False if they should be clipped to 0.
    """
    return bool(cfg.get("baseline", {}).get("allow_negative_baseline", False))


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

    Notes
    -----
    When ``baseline.allow_negative_baseline`` is ``True`` the corrected rates
    are preserved exactly as reported by the fit, even if they are negative.
    When it is ``False`` negative corrected rates are clipped to ``0.0`` so
    downstream consumers never observe negative activities without opting in.
    """

    baseline = cfg.get("baseline", {})
    tf = cfg.get("time_fit", {})
    scales = baseline.get("scales", {})
    base_rates = baseline.get("rate_Bq", {})
    corrected_rates = baseline.get("corrected_rate_Bq", {})
    if not isinstance(corrected_rates, Mapping):
        corrected_rates = {}

    allow_negative = baseline_allows_negative(cfg)

    summary: dict[str, tuple[float, float, float]] = {}
    for iso in isotopes:
        fit = tf.get(iso, {})
        raw = fit.get(f"E_{iso}")
        if raw is None:
            continue
        base = float(base_rates.get(iso, 0.0)) * float(scales.get(iso, 1.0))
        raw = float(raw)

        corr_val: float | None = None
        for candidate in (corrected_rates.get(iso), fit.get("E_corrected")):
            if candidate is None:
                continue
            try:
                corr_val = float(candidate)
            except (TypeError, ValueError):
                continue
            else:
                break

        corr = corr_val if corr_val is not None else raw - base
        if not allow_negative and corr < 0.0:
            corr = 0.0
        summary[iso] = (raw, base, corr)

    return summary
