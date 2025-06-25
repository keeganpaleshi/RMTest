import numpy as np
import pytest

from calibration import derive_calibration_constants


def test_fixed_slope_calibration():
    rng = np.random.default_rng(0)
    adc = rng.normal(1800, 2, 1000)

    cfg = {
        "calibration": {
            "slope_MeV_per_ch": 0.00435,
            "nominal_adc": {"Po214": 1800},
            "peak_search_radius": 10,
            "peak_prominence": 5,
            "peak_width": 1,
            "init_sigma_adc": 2.0,
        }
    }

    res = derive_calibration_constants(adc, cfg)
    idx = {exp: i for i, exp in enumerate(res._exponents)}
    slope = res.coeffs[idx[1]]

    assert slope == pytest.approx(0.00435)
