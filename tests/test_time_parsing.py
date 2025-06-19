import pytest
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze

@pytest.mark.parametrize("spec", ["0", "1970-01-01T00:00:00Z", "1970-01-01T00:00:00+00:00"])
def test_parse_time_arg_equivalent(spec):
    dt = analyze.parse_time_arg(spec)
    assert dt == datetime(1970, 1, 1, tzinfo=timezone.utc)
