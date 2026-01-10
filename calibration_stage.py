"""
calibration_stage.py

Energy calibration pipeline stage.

Handles energy calibration of ADC values to MeV using either:
- Two-point calibration from config
- Auto calibration from peaks

Also derives default time-fit windows from calibration peaks.
"""

import logging
import numpy as np
from typing import Any, Dict, Tuple
import pandas as pd

from calibration import (
    CalibrationResult,
    derive_calibration_constants,
    derive_calibration_constants_auto,
    DEFAULT_NOISE_CUTOFF,
)
from analysis_helpers import _roi_diff

logger = logging.getLogger(__name__)


def _value_sigma(val):
    """Extract value and sigma from various input formats."""
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) >= 2:
            return float(val[0]), float(val[1])
        if len(val) == 1:
            return float(val[0]), 0.0
        return 0.0, 0.0
    return float(val), 0.0


def _as_cal_result(obj):
    """Convert calibration dictionary to CalibrationResult object."""
    if isinstance(obj, CalibrationResult):
        return obj

    a, a_sig = _value_sigma(obj.get("a", 0.0))
    c, c_sig = _value_sigma(obj.get("c", 0.0))
    a2, a2_sig = _value_sigma(obj.get("a2", 0.0))
    sigma_E, sigma_E_error = _value_sigma(obj.get("sigma_E", 0.0))

    coeffs = [c, a]
    cov = np.array([[c_sig**2, 0.0], [0.0, a_sig**2]])

    if "ac_covariance" in obj:
        cov_ac = float(np.asarray(obj["ac_covariance"], dtype=float)[0][1])
        cov[0, 1] = cov[1, 0] = cov_ac

    if "a2" in obj:
        coeffs.append(a2)
        cov = np.pad(cov, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
        cov[2, 2] = a2_sig**2
        cov[1, 2] = cov[2, 1] = float(obj.get("cov_a_a2", 0.0))
        cov[0, 2] = cov[2, 0] = float(obj.get("cov_a2_c", 0.0))

    return CalibrationResult(
        coeffs=coeffs,
        cov=cov,
        sigma_E=sigma_E,
        sigma_E_error=sigma_E_error,
        peaks=obj.get("peaks"),
    )


def run_energy_calibration(
    df_analysis: pd.DataFrame,
    events_after_noise: pd.DataFrame,
    events_after_burst: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[CalibrationResult, Dict, pd.DataFrame, np.ndarray, np.ndarray, Dict, bool]:
    """
    Perform energy calibration and apply to dataframes.

    Returns:
        tuple: (cal_result, cal_params, df_analysis_updated, pre_spec_energies,
                post_spec_energies, roi_diff, calibration_valid)
    """
    adc_vals = df_analysis["adc"].values
    hist_bins = cfg["calibration"].get("hist_bins", 2000)
    calibration_valid = True

    try:
        if cfg.get("calibration", {}).get("method", "two-point") == "auto":
            adc_arr = df_analysis["adc"].to_numpy()
            cal_params = derive_calibration_constants_auto(
                adc_arr,
                noise_cutoff=cfg["calibration"].get(
                    "noise_cutoff", DEFAULT_NOISE_CUTOFF
                ),
                hist_bins=hist_bins,
                peak_search_radius=cfg["calibration"].get("peak_search_radius", 200),
                nominal_adc=cfg["calibration"].get("nominal_adc"),
            )
        else:
            # Two‐point calibration as given in config
            cal_params = derive_calibration_constants(adc_vals, config=cfg)
    except Exception:
        logging.exception("calibration failed – using defaults")
        if not cfg.get("allow_fallback"):
            raise
        calibration_valid = False
        cal_params = {"a": (0.005, 0.001), "c": (0.02, 0.005), "sigma_E": (0.3, 0.1)}

    cal_result = _as_cal_result(cal_params)

    # Apply calibration -> new column "energy_MeV" and its uncertainty
    energies = cal_result.predict(df_analysis["adc"])
    df_analysis["energy_MeV"] = energies
    df_analysis["denergy_MeV"] = cal_result.uncertainty(df_analysis["adc"])

    energies_pre_burst = cal_result.predict(events_after_noise["adc"])
    energies_post_burst = cal_result.predict(events_after_burst["adc"])
    roi_diff = _roi_diff(energies_pre_burst, energies_post_burst, cfg)
    pre_spec_energies = energies_pre_burst
    post_spec_energies = energies_post_burst

    # Derive default time-fit windows from calibration peaks when missing
    if getattr(cal_result, "peaks", None):
        tf_cfg = cfg.setdefault("time_fit", {})
        for iso in ("Po210", "Po218", "Po214"):
            key = f"window_{iso.lower()}"
            if tf_cfg.get(key) is None:
                peak_E = cal_result.peaks.get(iso, {}).get("centroid_mev")
                if peak_E is not None:
                    tf_cfg[key] = [float(peak_E - 0.08), float(peak_E + 0.08)]

    return (
        cal_result,
        cal_params,
        df_analysis,
        pre_spec_energies,
        post_spec_energies,
        roi_diff,
        calibration_valid,
    )
