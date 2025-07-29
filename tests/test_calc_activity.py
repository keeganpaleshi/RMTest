import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline


def test_calc_activity_basic():
    assert baseline.calc_activity(50, 10.0) == pytest.approx(5.0)


def test_calc_activity_zero_live_time():
    with pytest.raises(ValueError):
        baseline.calc_activity(1, 0)


def test_calc_activity_negative_live_time():
    with pytest.raises(ValueError):
        baseline.calc_activity(1, -1.0)


def test_calc_activity_negative_counts():
    with pytest.raises(ValueError):
        baseline.calc_activity(-5, 10.0)
