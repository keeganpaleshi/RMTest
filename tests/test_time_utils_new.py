import pandas as pd
from utils.time_utils import (
    to_datetime_utc,
    tz_localize_utc,
    tz_convert_utc,
    ensure_utc,
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
