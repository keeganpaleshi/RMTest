import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.time_utils import parse_timestamp


def test_parse_timestamp_iso_string():
    ts = parse_timestamp("1970-01-01T00:00:00Z")
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp("1970-01-01T00:00:00Z")


def test_parse_timestamp_numeric():
    ts = parse_timestamp(42)
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp(42, unit="s", tz="UTC")


def test_parse_timestamp_numeric_str():
    ts = parse_timestamp("42")
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp(42, unit="s", tz="UTC")


def test_parse_timestamp_naive_datetime():
    dt = datetime(1970, 1, 1)
    ts = parse_timestamp(dt)
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp("1970-01-01T00:00:00Z")


def test_parse_timestamp_float():
    ts = parse_timestamp(42.5)
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp(42.5, unit="s", tz="UTC")


def test_parse_timestamp_iso_without_tz():
    ts = parse_timestamp("1970-01-01T00:00:00")
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp("1970-01-01T00:00:00Z")


def test_parse_timestamp_iso_with_offset():
    ts = parse_timestamp("1970-01-01T01:00:00+01:00")
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp("1970-01-01T00:00:00Z")


def test_parse_timestamp_datetime_with_tz():
    dt = datetime(1970, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    ts = parse_timestamp(dt)
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp("1970-01-01T00:00:00Z")

