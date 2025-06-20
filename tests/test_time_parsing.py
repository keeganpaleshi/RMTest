import pytest
from datetime import datetime, timezone

from dateutil.tz import gettz
from utils import parse_time_arg

@pytest.mark.parametrize("inp", [
    0,
    "1970-01-01T00:00:00Z",
    "1970-01-01T00:00:00+00:00",
])
def test_parse_time_arg_variants(inp):
    dt = parse_time_arg(inp)
    assert dt == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_parse_time_arg_naive_with_zone():
    dt = parse_time_arg("1970-01-01T00:00:00", tz="US/Eastern")
    assert dt.tzinfo == gettz("US/Eastern")
    assert dt.timestamp() == pytest.approx(18000.0)
