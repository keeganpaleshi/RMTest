"""Time parsing helpers."""

import argparse
from datetime import datetime, timezone
from dateutil import parser as date_parser
from typing import Any, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional
    pd = None

from utils import parse_datetime

__all__ = ["parse_timestamp", "to_epoch_seconds"]


def parse_timestamp(value: Any) -> float:
    """Return Unix epoch seconds from ``value``.

    ``value`` may be a numeric value, ``datetime`` instance or an ISO-8601
    string. Strings lacking timezone information are interpreted as UTC.
    The result is always seconds since the Unix epoch in UTC.
    """

    if isinstance(value, (int, float)):
        return float(value)

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
            raise argparse.ArgumentTypeError(f"could not parse time: {value!r}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return float(dt.timestamp())

    raise argparse.ArgumentTypeError(f"could not parse time: {value!r}")


def to_epoch_seconds(value: Any) -> np.ndarray:
    """Convert timestamps to float seconds."""

    if pd is not None and isinstance(value, pd.Series):
        if not pd.api.types.is_datetime64_any_dtype(value):
            return value.astype(float).to_numpy()
        if getattr(value.dtype, "tz", None) is None:
            value = value.map(parse_datetime)
        else:
            value = value.dt.tz_convert("UTC")
        return value.astype("int64").to_numpy() / 1e9

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray([parse_timestamp(v) for v in value], dtype=float)

    return np.asarray([parse_timestamp(value)], dtype=float)
