import numpy as np
from calibration import derive_calibration_constants


def test_fixed_slope_calibration():
    rng = np.random.default_rng(0)
    adc = rng.normal(1800, 2, 1000)

    cfg = {
        "calibration": {
            "slope_MeV_per_ch": 0.00435,
            "nominal_adc": {"Po214": 1800},
            "peak_search_radius": 5,
            "peak_prominence": 0.0,
            "peak_width": 1,
            "init_sigma_adc": 5.0,
            "known_energies": {"Po214": 7.687},
        }
    }

    res = derive_calibration_constants(adc, cfg)
    assert res["a"] == 0.00435
    assert res["calibration_valid"] is True
