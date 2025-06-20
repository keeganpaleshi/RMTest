import sys
from pathlib import Path
import argparse
import pytest
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import parse_time_arg

@pytest.mark.parametrize("inp", [
    0,
    "1970-01-01T00:00:00Z",
    "1970-01-01T00:00:00+00:00",
])
def test_parse_time_arg_variants(inp):
    dt = parse_time_arg(inp)
    assert dt == datetime(1970, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("1970-01-01T00:00:00", datetime(1970, 1, 1, tzinfo=timezone.utc)),
        (
            "1970-01-01T01:00:00+01:00",
            datetime(1970, 1, 1, tzinfo=timezone.utc),
        ),
    ],
)
def test_parse_time_arg_iso_offsets(inp, expected):
    assert parse_time_arg(inp) == expected


@pytest.mark.parametrize(
    "inp,expected",
    [
        (0, datetime(1970, 1, 1, tzinfo=timezone.utc)),
        (0.5, datetime(1970, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)),
    ],
)
def test_parse_time_arg_numeric(inp, expected):
    assert parse_time_arg(inp) == expected


@pytest.mark.parametrize("inp", ["foo", "1970-13-01"])
def test_parse_time_arg_invalid(inp):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        parse_time_arg(inp)
