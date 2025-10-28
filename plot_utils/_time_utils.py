"""Internal helpers for time axis formatting and conversions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np


def _coerce_epoch_seconds(value):
    """Convert assorted time representations to epoch seconds."""

    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.datetime64):
        return float(value.astype("datetime64[s]").astype(np.int64))
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return float(value.timestamp())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                dt64 = np.datetime64(value)
            except ValueError:
                return None
            return float(dt64.astype("datetime64[s]").astype(np.int64))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return float(dt.timestamp())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_run_periods(
    periods: Sequence | None, t_start: float, t_end: float
) -> list[tuple[float, float]]:
    """Clamp and order run periods within ``[t_start, t_end]``."""

    if periods is None:
        periods = []

    normalized: list[tuple[float, float]] = []
    for item in periods:
        if not item:
            continue
        if isinstance(item, Mapping):
            start_raw = item.get("start")
            end_raw = item.get("end")
        else:
            try:
                start_raw, end_raw = item
            except (TypeError, ValueError):
                continue
        start = _coerce_epoch_seconds(start_raw)
        end = _coerce_epoch_seconds(end_raw)
        if start is None or end is None:
            continue
        start = max(float(t_start), float(start))
        end = min(float(t_end), float(end))
        if end <= start:
            continue
        normalized.append((start, end))

    if not normalized:
        normalized = [(float(t_start), float(t_end))]

    normalized.sort(key=lambda pair: pair[0])
    return normalized


def _cfg_lookup(cfg: Mapping | None, *keys, default=None):
    if not isinstance(cfg, Mapping):
        return default
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def build_time_series_segments(
    times: Iterable,
    t_start: float,
    t_end: float,
    cfg: Mapping | None = None,
    *,
    run_periods: Sequence | None = None,
):
    """Return bin definitions for time-series plots respecting run windows."""

    raw = np.asarray(times)
    if np.issubdtype(raw.dtype, np.datetime64):
        arr = raw.astype("datetime64[s]").astype(np.int64).astype(float)
    elif raw.dtype == object:
        arr = np.array([_coerce_epoch_seconds(val) for val in raw], dtype=float)
    else:
        arr = raw.astype(float)
    run_cfg = run_periods
    if run_cfg is None and isinstance(cfg, Mapping):
        run_cfg = _cfg_lookup(cfg, "analysis_run_periods")
    if run_cfg is None and isinstance(cfg, Mapping):
        run_cfg = _cfg_lookup(cfg, "run_periods")
    windows = normalize_run_periods(run_cfg, t_start, t_end)

    mask = np.zeros(arr.shape, dtype=bool)
    for start, end in windows:
        mask |= (arr >= start) & (arr <= end)

    bin_mode = str(
        _cfg_lookup(cfg, "plot_time_binning_mode", default=None)
        or _cfg_lookup(cfg, "time_bin_mode", default="fixed")
    ).lower()
    n_bins_fallback = int(_cfg_lookup(cfg, "time_bins_fallback", default=1) or 1)
    if n_bins_fallback < 1:
        n_bins_fallback = 1

    segments: list[dict[str, np.ndarray]] = []

    if bin_mode in ("fd", "auto"):
        data_rel = arr[mask] - float(t_start)
        bin_width = None
        if data_rel.size >= 2:
            q25, q75 = np.percentile(data_rel, [25, 75])
            iqr = q75 - q25
            if isinstance(iqr, np.timedelta64):
                iqr = float(iqr / np.timedelta64(1, "s"))
            iqr = float(iqr)
            if iqr > 0:
                width = 2 * iqr / (data_rel.size ** (1.0 / 3.0))
                if isinstance(width, np.timedelta64):
                    width = float(width / np.timedelta64(1, "s"))
                width = float(width)
                if np.isfinite(width) and width > 0:
                    bin_width = width

        for start, end in windows:
            if end <= start:
                continue
            if bin_width is not None:
                n_bins_run = max(1, int(np.ceil((end - start) / bin_width)))
            else:
                n_bins_run = n_bins_fallback
            edges_abs = np.linspace(float(start), float(end), n_bins_run + 1)
            centers_abs = 0.5 * (edges_abs[:-1] + edges_abs[1:])
            widths = np.diff(edges_abs)
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "edges": edges_abs,
                    "centers_abs": centers_abs,
                    "centers_rel": centers_abs - float(t_start),
                    "widths": widths,
                }
            )
    else:
        dt = float(_cfg_lookup(cfg, "plot_time_bin_width_s", default=None) or _cfg_lookup(cfg, "time_bin_s", default=3600))
        if dt <= 0:
            dt = 1.0
        for start, end in windows:
            if end <= start:
                continue
            run_range = end - start
            n_bins_run = int(np.floor(run_range / dt))
            if n_bins_run < 1:
                n_bins_run = 1
            edges_abs = float(start) + np.arange(0, (n_bins_run + 1) * dt, dt, dtype=float)
            centers_abs = 0.5 * (edges_abs[:-1] + edges_abs[1:])
            widths = np.diff(edges_abs)
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "edges": edges_abs,
                    "centers_abs": centers_abs,
                    "centers_rel": centers_abs - float(t_start),
                    "widths": widths,
                }
            )

    if not segments:
        edges_abs = np.array([float(t_start), float(t_end)], dtype=float)
        centers_abs = np.array([0.5 * (edges_abs[0] + edges_abs[1])], dtype=float)
        widths = np.diff(edges_abs)
        segments.append(
            {
                "start": float(t_start),
                "end": float(t_end),
                "edges": edges_abs,
                "centers_abs": centers_abs,
                "centers_rel": centers_abs - float(t_start),
                "widths": widths,
            }
        )

    return windows, segments


def to_mpl_times(times: Iterable) -> np.ndarray:
    """Convert an array-like of times to Matplotlib date numbers.

    Parameters
    ----------
    times : Iterable
        Sequence of epoch seconds, :class:`numpy.datetime64`, or
        :class:`datetime.datetime` objects. Naive datetimes are interpreted
        as UTC.

    Returns
    -------
    np.ndarray
        Array of floats suitable for Matplotlib time plotting.
    """

    arr = np.asarray(list(times))
    if np.issubdtype(arr.dtype, np.datetime64):
        secs = arr.astype("datetime64[s]").astype(np.int64).astype(float)
    elif arr.dtype == object:
        secs_list: list[float] = []
        for t in arr:
            if isinstance(t, datetime):
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                else:
                    t = t.astimezone(timezone.utc)
                secs_list.append(t.timestamp())
            elif isinstance(t, np.datetime64):
                secs_list.append(float(t.astype("datetime64[s]").astype(np.int64)))
            else:
                secs_list.append(float(t))
        secs = np.array(secs_list, dtype=float)
    else:
        secs = arr.astype(float)
    # mdates.epoch2num is not available in older Matplotlib versions,
    # so perform the conversion manually: seconds to days since 1970-01-01.
    epoch = mdates.date2num(datetime(1970, 1, 1))
    return secs / 86400.0 + epoch


def guard_mpl_times(*, times=None, times_mpl=None, times_dt=None) -> np.ndarray:
    """Normalize time inputs and forbid stale aliases.

    Exactly one of ``times`` or ``times_mpl`` must be supplied. The legacy
    ``times_dt`` alias is rejected to avoid accidental reintroduction of the
    old ``times_dt``/``times_mpl`` bug. The return value is always a NumPy
    array of Matplotlib date numbers.
    """

    if times_dt is not None:
        raise AssertionError("times_dt is deprecated; use 'times' or 'times_mpl'")

    provided = (times is not None, times_mpl is not None)
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of 'times' or 'times_mpl'")

    arr = times_mpl if times_mpl is not None else to_mpl_times(times)
    return np.asarray(arr, dtype=float)


def setup_time_axis(ax, times_mpl: np.ndarray):
    """Apply UTC date labels and elapsed-hour secondary axis."""
    locator = mdates.AutoDateLocator()
    try:  # Concise formatter is available on newer Matplotlib
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:  # pragma: no cover - fallback for old MPL
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    base = times_mpl[0]

    def _to_hours(x):
        return (x - base) * 24.0

    def _to_dates(x):
        return base + x / 24.0

    secax = ax.secondary_xaxis("top", functions=(_to_hours, _to_dates))
    secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos=None: f"{x:g}"))
    secax.set_xlabel("Elapsed Time (h)")

    ax.xaxis.get_offset_text().set_visible(False)
    secax.xaxis.get_offset_text().set_visible(False)
    return secax
