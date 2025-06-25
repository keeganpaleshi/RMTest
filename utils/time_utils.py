import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from datetime import datetime, timezone, tzinfo
from dateutil import parser as date_parser
from dateutil.tz import gettz

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


def parse_timestamp(value, tz="UTC") -> datetime:
    """Parse ``value`` to a UTC ``datetime`` object."""
    tzinfo_obj = tz if isinstance(tz, tzinfo) else gettz(tz)
    if tzinfo_obj is None:
        tzinfo_obj = timezone.utc

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo_obj)
        return dt.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if isinstance(value, str):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            pass
        try:
            dt = date_parser.isoparse(value)
        except (ValueError, OverflowError) as e:
            raise ValueError(f"could not parse timestamp: {value!r}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo_obj)
        return dt.astimezone(timezone.utc)

    raise ValueError(f"could not parse timestamp: {value!r}")


def to_epoch_seconds(value, tz="UTC") -> float:
    """Return Unix epoch seconds for ``value``."""
    return parse_timestamp(value, tz=tz).timestamp()

