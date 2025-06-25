import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

__all__ = [
    "to_datetime_utc",
    "tz_localize_utc",
    "tz_convert_utc",
    "ensure_utc",
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

