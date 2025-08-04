import numpy as np
import pytest

from calibration import (
    intercept_fit_two_point,
    CalibrationResult,
)


def sample_cfg():
    return {
        "calibration": {
            "slope_MeV_per_ch": 0.00435,
            "nominal_adc": {"Po210": 1246, "Po218": 1399, "Po214": 1800},
            "peak_search_radius": 5,
            "peak_prominence": 0.0,
            "peak_width": 1,
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 5.0,
            "known_energies": {"Po210": 5.304, "Po214": 7.687},
        }
    }


def test_intercept_fit_two_point_returns_expected_coeffs_and_covariance():
    rng = np.random.default_rng(42)
    adc = np.concatenate(
        [
            rng.normal(1246, 2, 300),
            rng.normal(1399, 2, 300),
            rng.normal(1800, 2, 300),
        ]
    )

    cfg = sample_cfg()
    res = intercept_fit_two_point(adc, cfg)

    a = cfg["calibration"]["slope_MeV_per_ch"]
    peaks = res.peaks

    c210 = 5.304 - a * peaks["Po210"]["centroid_adc"]
    c214 = 7.687 - a * peaks["Po214"]["centroid_adc"]
    expected_c = 0.5 * (c210 + c214)

    assert res.coeffs[1] == a
    assert res.coeffs[0] == pytest.approx(expected_c, abs=0.02)

    mu_err_210 = np.sqrt(peaks["Po210"]["covariance"][1][1])
    mu_err_214 = np.sqrt(peaks["Po214"]["covariance"][1][1])
    expected_var_c = (a ** 2 / 4.0) * (mu_err_210 ** 2 + mu_err_214 ** 2)
    assert res.cov[0, 0] == pytest.approx(expected_var_c, rel=1e-6)


def test_intercept_fit_two_point_missing_covariance(monkeypatch):
    def fake_calibrate_run(adc_values, cfg):
        slope = cfg["calibration"]["slope_MeV_per_ch"]
        peaks = {
            "Po210": {"centroid_adc": 1246.0},
            "Po214": {"centroid_adc": 1800.0, "sigma_adc": 2.0},
        }
        return CalibrationResult(coeffs=[0.0, slope], cov=np.zeros((2, 2)), peaks=peaks)

    monkeypatch.setattr("calibration.calibrate_run", fake_calibrate_run)

    adc = np.array([0.0])
    cfg = {"calibration": {"slope_MeV_per_ch": 0.00435}}
    with pytest.raises(KeyError):
        intercept_fit_two_point(adc, cfg)


def test_intercept_fit_two_point_missing_slope():
    adc = np.array([0.0])
    cfg = {"calibration": {}}
    with pytest.raises(KeyError):
        intercept_fit_two_point(adc, cfg)
