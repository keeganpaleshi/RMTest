import numpy as np
from calibration import calibrate_run, CalibrationResult
from constants import DEFAULT_KNOWN_ENERGIES


def intercept_fit_two_point(adc_values, cfg):
    """Return calibration with fixed slope using Po-210 and Po-214 anchors."""
    a = cfg["calibration"]["slope_MeV_per_ch"]
    # Use full calibration routine to locate peaks for both isotopes
    cal_res = calibrate_run(adc_values, cfg)

    energies = {**DEFAULT_KNOWN_ENERGIES, **cfg.get("calibration", {}).get("known_energies", {})}
    adc210 = cal_res.peaks["Po210"]["centroid_adc"]
    adc214 = cal_res.peaks["Po214"]["centroid_adc"]
    c210 = energies["Po210"] - a * adc210
    c214 = energies["Po214"] - a * adc214
    c = 0.5 * (c210 + c214)

    mu_err_210 = float(np.sqrt(cal_res.peaks["Po210"]["covariance"][1][1]))
    mu_err_214 = float(np.sqrt(cal_res.peaks["Po214"]["covariance"][1][1]))
    var_c = (a ** 2 / 4.0) * (mu_err_210 ** 2 + mu_err_214 ** 2)
    cov = np.array([[var_c, 0.0], [0.0, 0.0]])

    sigma_adc = cal_res.peaks["Po214"]["sigma_adc"]
    dsigma_adc = float(np.sqrt(cal_res.peaks["Po214"]["covariance"][2][2]))
    sigma_E = abs(a) * sigma_adc
    dsigma_E = abs(a) * dsigma_adc

    return CalibrationResult(
        coeffs=[c, a],
        cov=cov,
        peaks=cal_res.peaks,
        sigma_E=float(sigma_E),
        sigma_E_error=float(dsigma_E),
    )


__all__ = ["intercept_fit_two_point"]
