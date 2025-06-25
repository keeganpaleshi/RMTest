import pandas as pd
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



def parse_timestamp(value):
    """Return ``value`` parsed to a UTC ``pandas.Timestamp``."""
    from datetime import datetime, timezone
    from dateutil import parser as date_parser

    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return pd.Timestamp(dt)

    if isinstance(value, (int, float)):
        return pd.to_datetime(float(value), unit="s", utc=True)

    if isinstance(value, str):
        try:
            return pd.to_datetime(float(value), unit="s", utc=True)
        except ValueError:
            pass
        try:
            dt = date_parser.isoparse(value)
        except (ValueError, OverflowError) as e:
            raise ValueError(f"invalid timestamp: {value!r}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return pd.Timestamp(dt)

    raise ValueError(f"invalid timestamp: {value!r}")


def to_epoch_seconds(ts_or_str) -> float:
    """Return Unix epoch seconds for ``ts_or_str``."""

    ts = parse_timestamp(ts_or_str)
    return ts.timestamp()

