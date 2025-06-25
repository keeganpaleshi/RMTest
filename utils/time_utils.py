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


def parse_timestamp(value) -> float:
    """Parse an ISO-8601 string, numeric seconds or datetime-like object."""

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        return float(value.timestamp())

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return float(dt.timestamp())

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        try:
            dt = date_parser.isoparse(value)
        except (ValueError, OverflowError) as e:
            raise ValueError(f"could not parse time: {value!r}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return float(dt.timestamp())

    raise ValueError(f"could not parse time: {value!r}")


def to_epoch_seconds(value) -> float:
    """Return ``value`` converted to Unix epoch seconds."""

    return parse_timestamp(value)

