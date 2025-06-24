import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import parse_datetime


def test_parse_datetime_iso_string():
    ts = parse_datetime("1970-01-01T00:00:00Z")
    assert isinstance(ts, np.datetime64)
    assert ts == np.datetime64("1970-01-01T00:00:00")


def test_parse_datetime_numeric():
    ts = parse_datetime(42)
    assert isinstance(ts, np.datetime64)
    assert ts == np.datetime64(42, "s")


def test_parse_datetime_numeric_str():
    ts = parse_datetime("42")
    assert isinstance(ts, np.datetime64)
    assert ts == np.datetime64(42, "s")


def test_parse_datetime_naive_datetime():
    dt = datetime(1970, 1, 1)
    ts = parse_datetime(dt)
    assert isinstance(ts, np.datetime64)
    assert ts == np.datetime64("1970-01-01T00:00:00")


def test_parse_datetime_float():
    ts = parse_datetime(42.5)
    assert isinstance(ts, np.datetime64)
    assert ts == np.datetime64("1970-01-01T00:00:42.500000000")


def test_parse_datetime_iso_without_tz():
    ts = parse_datetime("1970-01-01T00:00:00")
    assert isinstance(ts, np.datetime64)
    assert ts == np.datetime64("1970-01-01T00:00:00")


def test_parse_datetime_iso_with_offset():
    ts = parse_datetime("1970-01-01T01:00:00+01:00")
    assert isinstance(ts, np.datetime64)
    assert ts == np.datetime64("1970-01-01T00:00:00")


def test_parse_datetime_datetime_with_tz():
    dt = datetime(1970, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    ts = parse_datetime(dt)
    assert isinstance(ts, np.datetime64)
    assert ts == np.datetime64("1970-01-01T00:00:00")

