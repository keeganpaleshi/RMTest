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
            "peak_search_radius": 5,
            "peak_prominence": 0.0,
            "peak_width": 1,
            "init_sigma_adc": 5.0,
            "known_energies": {"Po214": 7.687},
        }
    }

    res = derive_calibration_constants(adc, cfg)
    assert res.coeffs[1] == 0.00435
    assert res.coeffs[0] == pytest.approx(-0.14, abs=0.02)
    assert res.sigma_E == pytest.approx(0.00435 * 2, rel=0.2)


def test_float_slope_calibration():
    rng = np.random.default_rng(1)
    adc = np.concatenate(
        [
            rng.normal(1242, 2, 200),
            rng.normal(1405, 2, 200),
            rng.normal(1800, 2, 200),
        ]
    )

    cfg = {
        "calibration": {
            "slope_MeV_per_ch": 0.004,
            "float_slope": True,
            "nominal_adc": {"Po210": 0, "Po218": 0, "Po214": 0},
            "peak_search_radius": 200,
            "peak_prominence": 0.0,
            "peak_width": 1,
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 4.0,
            "init_tau_adc": 0.0,
            "known_energies": {
                "Po210": 5.304,
                "Po218": 6.002,
                "Po214": 7.687,
            },
            "sanity_tolerance_mev": 1.0,
        }
    }

    res = derive_calibration_constants(adc, cfg)
    assert res.coeffs[1] == pytest.approx(0.00427, rel=0.05)
