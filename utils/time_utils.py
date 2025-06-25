from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from datetime import datetime, timezone
from dateutil import parser as date_parser

__all__ = [
    "to_datetime_utc",
    "tz_localize_utc",
    "tz_convert_utc",
    "ensure_utc",
    "parse_timestamp",
    "to_epoch_seconds",
]


def to_datetime_utc(values, *args, **kwargs):
    """Wrapper for :func:`pandas.to_datetime` with ``utc=True``."""
    kwargs.setdefault("utc", True)
    return pd.to_datetime(values, *args, **kwargs)


def tz_localize_utc(series: pd.Series):
    """Localize naive datetime ``Series`` to UTC."""
    if not is_datetime64_any_dtype(series):
        raise TypeError("tz_localize_utc expects datetime64 dtype")
    if getattr(series.dtype, "tz", None) is not None:
        return series
    return series.dt.tz_localize("UTC")


def tz_convert_utc(series: pd.Series):
    """Convert timezone-aware ``Series`` to UTC."""
    if not is_datetime64_any_dtype(series):
        raise TypeError("tz_convert_utc expects datetime64 dtype")
    if getattr(series.dtype, "tz", None) is None:
        return series.dt.tz_localize("UTC")
    return series.dt.tz_convert("UTC")


def ensure_utc(series: pd.Series) -> pd.Series:
    """Return ``series`` converted or localized to UTC."""
    if getattr(series.dtype, "tz", None) is None:
        return tz_localize_utc(series)
    return tz_convert_utc(series)


def parse_timestamp(value: str | int | float | datetime) -> pd.Timestamp:
    """Return ``value`` parsed as a UTC ``Timestamp``."""

    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, (int, float)):
        ts = pd.Timestamp(float(value), unit="s", tz="UTC")
    elif isinstance(value, datetime):
        if value.tzinfo is None:
            ts = pd.Timestamp(value, tz="UTC")
        else:
            ts = pd.Timestamp(value).tz_convert("UTC")
    elif isinstance(value, str):
        try:
            ts = pd.Timestamp(float(value), unit="s", tz="UTC")
        except ValueError:
            dt = date_parser.isoparse(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            ts = pd.Timestamp(dt)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def to_epoch_seconds(ts: pd.Timestamp | str | int | float) -> float:
    """Return ``ts`` converted to Unix seconds (UTC)."""

    if not isinstance(ts, pd.Timestamp):
        ts = parse_timestamp(ts)
    elif ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.value / 1e9

