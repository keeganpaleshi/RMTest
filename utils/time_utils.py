import pandas as pd
from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype

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


def parse_timestamp(value) -> pd.Timestamp:
    """Return ``value`` parsed to a UTC ``pandas.Timestamp``."""

    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, (int, float)):
        ts = pd.Timestamp(float(value), unit="s", tz="UTC")
    elif isinstance(value, str):
        try:
            ts = pd.Timestamp(float(value), unit="s", tz="UTC")
        except ValueError:
            ts = pd.to_datetime(value, utc=True)
    else:
        raise TypeError(f"cannot parse timestamp from {value!r}")

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def to_epoch_seconds(ts) -> float:
    """Convert ``ts`` to Unix epoch seconds as ``float``."""

    if isinstance(ts, pd.Timestamp):
        t = ts
    else:
        t = parse_timestamp(ts)

    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.value / 1e9

