import pytest
from datetime import datetime, timezone

from utils import parse_time_arg

@pytest.mark.parametrize("inp", [
    0,
    "1970-01-01T00:00:00Z",
    "1970-01-01T00:00:00+00:00",
])
def test_parse_time_arg_variants(inp):
    dt = parse_time_arg(inp)
    assert dt == datetime(1970, 1, 1, tzinfo=timezone.utc)
