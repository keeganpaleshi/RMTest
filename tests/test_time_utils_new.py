import pandas as pd
import pytest
from utils.time_utils import (
    to_datetime_utc,
    tz_localize_utc,
    tz_convert_utc,
    ensure_utc,
    parse_timestamp,
    to_epoch_seconds,
)


def test_to_datetime_utc():
    out = to_datetime_utc([0, 1], unit="s")
    assert isinstance(out, pd.DatetimeIndex)
    assert str(out.dtype) == "datetime64[ns, UTC]"


def test_tz_localize_and_convert():
    ser_naive = pd.Series(pd.date_range("1970-01-01", periods=2, freq="s"))
    localized = tz_localize_utc(ser_naive)
    assert str(localized.dtype) == "datetime64[ns, UTC]"

    ser_aware = pd.Series(pd.date_range("1970-01-01", periods=2, freq="s", tz="US/Eastern"))
    converted = tz_convert_utc(ser_aware)
    assert str(converted.dtype) == "datetime64[ns, UTC]"


def test_ensure_utc():
    ser_naive = pd.Series(pd.date_range("1970-01-01", periods=2, freq="s"))
    ser_aware = pd.Series(pd.date_range("1970-01-01", periods=2, freq="s", tz="US/Eastern"))
    assert str(ensure_utc(ser_naive).dtype) == "datetime64[ns, UTC]"
    assert str(ensure_utc(ser_aware).dtype) == "datetime64[ns, UTC]"


def test_parse_timestamp_and_epoch():
    ts = parse_timestamp("1970-01-01T00:00:01Z")
    assert isinstance(ts, pd.Timestamp)
    assert str(ts.tz) == "UTC"
    assert to_epoch_seconds(ts) == pytest.approx(1.0)
