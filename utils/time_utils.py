import pandas as pd
from datetime import datetime
from dateutil import parser as date_parser

__all__ = ["parse_timestamp", "to_epoch_seconds"]


def _is_numeric_string(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_timestamp(value: str | int | float | datetime) -> pd.Timestamp:
    """Return ``value`` converted to a UTC ``pandas.Timestamp``."""

    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, (int, float)) or (isinstance(value, str) and _is_numeric_string(value)):
        ts = pd.to_datetime(float(value), unit="s", utc=True)
    else:
        try:
            ts = pd.to_datetime(value, utc=True)
        except Exception as e:
            try:
                ts = pd.Timestamp(date_parser.isoparse(str(value)))
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            except Exception:
                raise ValueError(f"invalid timestamp: {value!r}") from e
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def to_epoch_seconds(ts: pd.Timestamp | str | int | float) -> float:
    """Return Unix epoch seconds for ``ts``."""

    if isinstance(ts, (int, float)):
        return float(ts)
    return float(parse_timestamp(ts).timestamp())

