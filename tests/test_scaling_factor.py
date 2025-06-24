import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_utils as baseline


def test_scaling_factor_basic():
    s, ds = baseline.scaling_factor(10.0, 5.0)
    assert s == pytest.approx(2.0)
    assert ds == pytest.approx(0.0)


def test_scaling_factor_uncertainty():
    s, ds = baseline.scaling_factor(10.0, 5.0, 0.1, 0.2)
    assert s == pytest.approx(2.0)
    expected_var = (0.1/5.0)**2 + ((10.0*0.2)/5.0**2)**2
    assert ds == pytest.approx((expected_var)**0.5)


def test_scaling_factor_zero_baseline():
    with pytest.raises(ValueError):
        baseline.scaling_factor(1.0, 0.0)
