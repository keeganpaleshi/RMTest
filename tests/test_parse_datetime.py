import sys
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import parse_datetime


def test_parse_datetime_iso_string():
    ts = parse_datetime("1970-01-01T00:00:00Z")
    assert ts == np.datetime64("1970-01-01T00:00:00Z")


def test_parse_datetime_numeric():
    ts = parse_datetime(42)
    assert ts == np.datetime64("1970-01-01T00:00:42Z")


def test_parse_datetime_numeric_str():
    ts = parse_datetime("42")
    assert ts == np.datetime64("1970-01-01T00:00:42Z")


def test_parse_datetime_naive_datetime():
    dt = datetime(1970, 1, 1)
    ts = parse_datetime(dt)
    assert ts == np.datetime64("1970-01-01T00:00:00Z")

