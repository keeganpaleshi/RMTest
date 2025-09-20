from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_utils
from radon.baseline import _scaling_factor


def test_scaling_factor_basic():
    s, ds = _scaling_factor(10.0, 5.0)
    assert s == pytest.approx(2.0)
    assert ds == pytest.approx(0.0)


def test_scaling_factor_uncertainty():
    s, ds = _scaling_factor(10.0, 5.0, 0.1, 0.2)
    assert s == pytest.approx(2.0)
    expected_var = (0.1/5.0)**2 + ((10.0*0.2)/5.0**2)**2
    assert ds == pytest.approx((expected_var)**0.5)


def test_scaling_factor_zero_baseline():
    with pytest.raises(ValueError):
        _scaling_factor(1.0, 0.0)


def test_compute_dilution_factor():
    d = baseline_utils.compute_dilution_factor(10.0, 5.0)
    assert d == pytest.approx(10.0 / 15.0)


@pytest.mark.parametrize(
    "monitor_volume, sample_volume, message",
    [
        (0.0, 5.0, "monitor_volume must be positive"),
        (-1.0, 5.0, "monitor_volume must be positive"),
        (10.0, -0.5, "sample_volume must be non-negative"),
        (math.nan, 1.0, "monitor_volume must be finite"),
        (math.inf, 1.0, "monitor_volume must be finite"),
        (10.0, math.nan, "sample_volume must be finite"),
        (10.0, math.inf, "sample_volume must be finite"),
    ],
)
def test_compute_dilution_factor_invalid_inputs(monitor_volume, sample_volume, message):
    with pytest.raises(ValueError) as exc:
        baseline_utils.compute_dilution_factor(monitor_volume, sample_volume)
    assert message in str(exc.value)
