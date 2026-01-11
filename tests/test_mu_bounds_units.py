import numpy as np
import pytest

from analyze import _normalise_mu_bounds
from calibration import apply_calibration


def test_normalise_mu_bounds_mev_passthrough():
    bounds_cfg = {"Po210": [5.2, 5.4]}
    out = _normalise_mu_bounds(
        bounds_cfg,
        units="mev",
        slope=0.005,
        intercept=0.02,
        quadratic_coeff=0.0,
    )

    assert out == {"Po210": (5.2, 5.4)}


def test_normalise_mu_bounds_adc_conversion():
    slope = 0.005
    intercept = 0.02
    bounds_adc = {"Po210": [1000, 1010]}

    out = _normalise_mu_bounds(
        bounds_adc,
        units="adc",
        slope=slope,
        intercept=intercept,
        quadratic_coeff=0.0,
    )

    expected = apply_calibration([1000, 1010], slope, intercept)
    assert np.allclose(out["Po210"], expected)


def test_normalise_mu_bounds_invalid_units():
    with pytest.raises(ValueError):
        _normalise_mu_bounds(
            {"Po210": [5.2, 5.4]},
            units="kev",
            slope=0.005,
            intercept=0.02,
            quadratic_coeff=0.0,
        )


def test_normalise_mu_bounds_requires_strict_order():
    with pytest.raises(ValueError):
        _normalise_mu_bounds(
            {"Po210": [5.4, 5.4]},
            units="mev",
            slope=0.005,
            intercept=0.02,
            quadratic_coeff=0.0,
        )
