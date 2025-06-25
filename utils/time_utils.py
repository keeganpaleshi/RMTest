"""Time conversion utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Union

import pandas as pd
from dateutil import parser as date_parser

__all__ = ["parse_timestamp", "to_epoch_seconds"]


TimestampLike = Union[str, int, float, datetime, pd.Timestamp]


def parse_timestamp(value: TimestampLike) -> pd.Timestamp:
    """Return ``value`` parsed as a timezone-aware ``pandas.Timestamp`` in UTC."""
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, (int, float)):
        ts = pd.Timestamp(float(value), unit="s", tz="UTC")
    elif isinstance(value, str):
        try:
            num = float(value)
            ts = pd.Timestamp(num, unit="s", tz="UTC")
        except ValueError:
            try:
                ts = pd.Timestamp(date_parser.isoparse(value))
            except Exception as e:
                raise ValueError(f"invalid timestamp: {value!r}") from e
    else:
        raise ValueError(f"invalid timestamp: {value!r}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def to_epoch_seconds(ts_or_str: TimestampLike) -> float:
    """Convert ``ts_or_str`` to Unix epoch seconds."""
    ts = parse_timestamp(ts_or_str)
    return ts.timestamp()
