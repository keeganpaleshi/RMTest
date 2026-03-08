from datetime import datetime, timezone, tzinfo
from typing import Any, Union

import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from dateutil.tz import gettz
from pandas.api.types import is_datetime64_any_dtype

__all__ = [
    "to_datetime_utc",
    "tz_localize_utc",
    "tz_convert_utc",
    "ensure_utc",
    "to_utc_datetime",
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


def _resolve_tzinfo(tz: str | tzinfo | None = "UTC") -> tzinfo:
    """Return a timezone object, defaulting to UTC for unknown values."""

    if isinstance(tz, tzinfo):
        return tz
    resolved = gettz(tz) if tz is not None else None
    return resolved or timezone.utc


def to_utc_datetime(value: Any, tz: str | tzinfo = "UTC") -> datetime:
    """Return ``value`` converted to a timezone-aware UTC ``datetime``."""

    tzinfo_obj = _resolve_tzinfo(tz)

    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.to_pydatetime()

    if isinstance(value, np.datetime64):
        return parse_timestamp(value, tz=tzinfo_obj).to_pydatetime()

    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=tzinfo_obj)
        return dt.astimezone(timezone.utc)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if isinstance(value, str):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            pass

        try:
            dt = date_parser.isoparse(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"invalid datetime: {value!r}") from exc

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo_obj)
        return dt.astimezone(timezone.utc)

    raise ValueError(f"invalid datetime: {value!r}")


def parse_timestamp(
    value: Union[str, int, float, datetime, pd.Timestamp, np.datetime64],
    tz: str | tzinfo = "UTC",
) -> pd.Timestamp:
    """Return ``value`` parsed to a UTC :class:`pandas.Timestamp`."""

    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    if isinstance(value, np.datetime64):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(_resolve_tzinfo(tz))
        return ts.tz_convert("UTC")

    if isinstance(value, (datetime, int, float, np.integer, np.floating)):
        return pd.Timestamp(to_utc_datetime(value, tz=tz))

    if isinstance(value, str):
        try:
            return pd.to_datetime(float(value), unit="s", utc=True)
        except ValueError:
            pass
        try:
            return pd.Timestamp(to_utc_datetime(value, tz=tz))
        except ValueError as exc:
            raise ValueError(f"invalid timestamp: {value!r}") from exc

    raise ValueError(f"invalid timestamp: {value!r}")


def to_epoch_seconds(
    ts: Union[pd.Timestamp, str, int, float, datetime, np.datetime64],
    tz: str | tzinfo = "UTC",
) -> float:
    """Return Unix epoch seconds for ``ts``."""

    ts_parsed = parse_timestamp(ts, tz=tz)
    return ts_parsed.timestamp()

