from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from utils.time_utils import (
    ensure_utc,
    parse_timestamp,
    to_datetime_utc,
    to_epoch_seconds,
    to_utc_datetime,
    tz_convert_utc,
    tz_localize_utc,
)


def _is_utc_datetime(dtype):
    """Check if dtype is a UTC-aware datetime64 regardless of resolution."""
    s = str(dtype)
    return s.startswith("datetime64[") and "UTC" in s


def test_to_datetime_utc():
    out = to_datetime_utc([0, 1], unit="s")
    assert isinstance(out, pd.DatetimeIndex)
    assert _is_utc_datetime(out.dtype)


def test_tz_localize_and_convert():
    ser_naive = pd.Series(pd.date_range("1970-01-01", periods=2, freq="s"))
    localized = tz_localize_utc(ser_naive)
    assert _is_utc_datetime(localized.dtype)

    ser_aware = pd.Series(pd.date_range("1970-01-01", periods=2, freq="s", tz="America/New_York"))
    converted = tz_convert_utc(ser_aware)
    assert _is_utc_datetime(converted.dtype)


def test_ensure_utc():
    ser_naive = pd.Series(pd.date_range("1970-01-01", periods=2, freq="s"))
    ser_aware = pd.Series(pd.date_range("1970-01-01", periods=2, freq="s", tz="America/New_York"))
    assert _is_utc_datetime(ensure_utc(ser_naive).dtype)
    assert _is_utc_datetime(ensure_utc(ser_aware).dtype)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, pd.Timestamp(0, unit="s", tz="UTC")),
        (0.5, pd.Timestamp(0.5, unit="s", tz="UTC")),
        ("1970-01-01T00:00:00Z", pd.Timestamp(0, unit="s", tz="UTC")),
        (datetime(1970, 1, 1), pd.Timestamp(0, unit="s", tz="UTC")),
        (
            datetime(1970, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))),
            pd.Timestamp(0, unit="s", tz="UTC"),
        ),
    ],
)
def test_parse_timestamp_variants(value, expected):
    assert parse_timestamp(value) == expected


def test_parse_timestamp_naive_inputs_respect_timezone_override():
    expected = pd.Timestamp(0, unit="s", tz="UTC")
    assert parse_timestamp("1970-01-01T01:00:00", tz="Europe/Berlin") == expected
    assert parse_timestamp(datetime(1970, 1, 1, 1), tz="Europe/Berlin") == expected


def test_parse_timestamp_numpy_datetime64():
    assert parse_timestamp(np.datetime64("1970-01-01T00:00:01")) == pd.Timestamp(1, unit="s", tz="UTC")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (42.5, 42.5),
        ("42", 42.0),
        ("1970-01-01T00:00:00.5Z", 0.5),
        (datetime(1970, 1, 1, tzinfo=timezone.utc), 0.0),
        (np.datetime64("1970-01-01T00:00:01"), 1.0),
    ],
)
def test_to_epoch_seconds_variants(value, expected):
    assert to_epoch_seconds(value) == pytest.approx(expected)


def test_to_utc_datetime_naive_inputs_respect_timezone_override():
    expected = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert to_utc_datetime("1970-01-01T01:00:00", tz="Europe/Berlin") == expected
    assert to_utc_datetime(datetime(1970, 1, 1, 1), tz="Europe/Berlin") == expected

