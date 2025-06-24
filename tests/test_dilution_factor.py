import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_utils as bu


def test_dilution_factor_basic():
    assert bu.dilution_factor(10.0, 10.0) == pytest.approx(0.5)
    assert bu.dilution_factor(10.0, 0.0) == pytest.approx(1.0)
    assert bu.dilution_factor(0.0, 10.0) == pytest.approx(0.0)


def test_dilution_factor_zero_total():
    assert bu.dilution_factor(0.0, 0.0) == 0.0
