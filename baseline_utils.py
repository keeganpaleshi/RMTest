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


class BaselineError(Exception):
    """Raised when baseline diagnostics fail."""


def summarize_baseline(cfg: dict, isotopes: list[str]) -> dict[str, tuple[float, float, float]]:
    """Return baseline diagnostics for *isotopes*.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing ``baseline`` and ``time_fit`` sections.
    isotopes : list of str
        Isotopes to include in the summary.

    Returns
    -------
    dict
        Mapping of isotope name to ``(raw_rate, baseline_rate, corrected_rate)`` tuples.
    """

    bl_cfg = cfg.get("baseline")
    if not bl_cfg:
        raise BaselineError("baseline configuration missing")

    paths = bl_cfg.get("files")
    if not paths:
        raise BaselineError("baseline.files not specified")
    if isinstance(paths, str):
        paths = [paths]

    rng = bl_cfg.get("range")
    if not rng or len(rng) != 2:
        raise BaselineError("baseline.range must contain [start, end]")
    start, end = rng

    monitor_vol = float(bl_cfg.get("monitor_volume_l", 0.0))
    sample_vol = float(bl_cfg.get("sample_volume_l", 0.0))
    dilution = compute_dilution_factor(monitor_vol, sample_vol)

    events = []
    for p in paths:
        df = load_events(p, start=start, end=end)
        if not df.empty:
            events.append(df)
    if not events:
        raise BaselineError("no baseline events")
    df_bl = pd.concat(events, ignore_index=True)

    ts = df_bl["timestamp"].astype("int64")
    live = float((ts.max() - ts.min()) / 1e9)
    if live <= 0:
        raise BaselineError("zero live-time in baseline range")

    results = {}
    for iso in isotopes:
        win = cfg.get("time_fit", {}).get(f"window_{iso.lower()}")
        if not win:
            continue
        lo, hi = win
        mask = (df_bl.get("energy_MeV", df_bl["adc"]) >= lo) & (
            df_bl.get("energy_MeV", df_bl["adc"]) <= hi
        )
        counts = int(mask.sum())
        raw_rate = counts / live if live > 0 else 0.0
        baseline_rate = raw_rate * dilution
        results[iso] = (raw_rate, baseline_rate, raw_rate - baseline_rate)

    return results

__all__ = [
    "BaselineError",
    "summarize_baseline",
    "compute_dilution_factor",
    "subtract_baseline_dataframe",
    "subtract_baseline_counts",
    "subtract_baseline_rate",
    "rate_histogram",
    "summarize_baseline",
    "BaselineError",
]


class BaselineError(RuntimeError):
    """Raised when baseline subtraction yields negative rates."""

    pass




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
) -> pd.DataFrame:
    """Return new ``DataFrame`` with baseline-subtracted spectra.

    ``t_base0`` and ``t_base1`` may be naïve or timezone-aware. They are
    interpreted in UTC and compared using integer nanoseconds to avoid
    issues with differing time zone information between ``df_full`` and
    the provided range. When ``allow_fallback`` is ``False`` (default), a
    :class:`RuntimeError` is raised if the baseline range contains no
    events.
    """

    assert mode in ("none", "electronics", "radon", "all")

    if mode == "none":
        return df_analysis.copy()

    rate_an, live_an = rate_histogram(df_analysis, bins)
    if live_time_analysis is None:
        live_time_analysis = live_an

    t0 = parse_timestamp(t_base0)
    t1 = parse_timestamp(t_base1)
    ts_full = _to_datetime64(df_full)
    ts_int = ts_full.view("int64")
    t0_ns = t0.value
    t1_ns = t1.value
    mask = (ts_int >= t0_ns) & (ts_int <= t1_ns)
    if not mask.any():
        msg = "baseline_range matched no events – skipping subtraction"
        if not allow_fallback:
            raise RuntimeError(msg)
        logging.warning(msg)
        return df_analysis.copy()

    rate_bl, live_bl = rate_histogram(df_full.loc[mask], bins)

    if mode in ("electronics", "radon", "all"):
        net_counts = (rate_an - rate_bl) * live_time_analysis
    else:  # mode == "none"
        net_counts = rate_an * live_time_analysis

    df_out = df_analysis.copy()
    df_out["subtracted_adc_hist"] = [net_counts] * len(df_out)
    return df_out


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return baseline-corrected spectra and statistical errors.

    Uncertainties are propagated in quadrature from the analysis and
    baseline histograms unless ``kw.get("uncert_prop") == "none"``.
    If ``allow_fallback`` is ``False`` (default) a
    :class:`RuntimeError` is raised when the baseline interval contains
    no events.
    """

    df_corr = apply_baseline_subtraction(
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

    df_err = df_corr.copy()
    df_err["subtracted_adc_hist"] = [err_hist] * len(df_err)

    return df_corr, df_err


# Backwards compatibility
subtract_baseline_dataframe = apply_baseline_subtraction



# thin wrappers are imported above from :mod:`radon.baseline`


def summarize_baseline(cfg: dict, isotopes: list[str]) -> dict[str, tuple[float, float, float]]:
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

