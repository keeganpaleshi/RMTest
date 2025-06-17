import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import (
    two_point_calibration,
    apply_calibration,
    emg_left,
    gaussian,
    DEFAULT_KNOWN_ENERGIES,
    derive_calibration_constants,
)


def test_two_point_calibration():
    # Synthetic ADC peaks
    peak_adc = {"Po210": 1000, "Po214": 2000, "Po218": 1500}
    known = {"Po210": 5.30, "Po214": 7.69, "Po218": 6.00}
    m, c = two_point_calibration(
        [peak_adc["Po210"], peak_adc["Po214"]],
        [known["Po210"], known["Po214"]],
    )
    # Check that calibration maps 1000->5.30 and 2000->7.69
    assert pytest.approx(m * 1000 + c, rel=1e-3) == 5.30
    assert pytest.approx(m * 2000 + c, rel=1e-3) == 7.69


def test_two_point_calibration_identical_centroids_error():
    with pytest.raises(ValueError):
        two_point_calibration([1000, 1000], [5.3, 7.7])


def test_apply_calibration():
    slope, intercept = 0.005, 0.02
    adc_vals = np.array([0, 100, 200])
    energies = apply_calibration(adc_vals, slope, intercept)
    assert np.allclose(energies, np.array([0.02, 0.52, 1.02]))


def test_emg_left_finite_near_zero_tau():
    """emg_left should remain finite as tau approaches zero."""
    x = np.linspace(-2.0, 2.0, 5)
    mu, sigma = 0.0, 1.0

    pdf_gauss = gaussian(x, mu, sigma)
    pdf_small_tau = emg_left(x, mu, sigma, 1e-4)

    assert np.all(np.isfinite(pdf_small_tau))
    assert np.allclose(pdf_small_tau, pdf_gauss, rtol=2e-4)


def test_derive_calibration_constants_peak_search_radius():
    """derive_calibration_constants should honor peak_search_radius."""
    rng = np.random.default_rng(0)
    # Generate peaks slightly offset from the nominal ADC guesses
    adc = np.concatenate(
        [
            rng.normal(803, 2, 500),
            rng.normal(1004, 2, 500),
            rng.normal(1197, 2, 500),
        ]
    )

    base_cfg = {
        "calibration": {
            "peak_prominence": 5,
            "peak_width": 1,
            "nominal_adc": {"Po210": 800, "Po218": 1000, "Po214": 1200},
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 4.0,
            "init_tau_adc": 0.0,
            "sanity_tolerance_mev": 1.0,
        }
    }

    cfg_ok = {"calibration": dict(base_cfg["calibration"])}
    cfg_ok["calibration"]["peak_search_radius"] = 5

    out = derive_calibration_constants(adc, cfg_ok)
    assert set(out["peaks"].keys()) == {"Po210", "Po218", "Po214"}

    cfg_bad = {"calibration": dict(base_cfg["calibration"])}
    cfg_bad["calibration"]["peak_search_radius"] = 1

    with pytest.raises(RuntimeError):
        derive_calibration_constants(adc, cfg_bad)


def test_calibration_uses_known_energies_from_config():
    """Custom energies in config should override defaults."""
    rng = np.random.default_rng(1)
    adc = np.concatenate([
        rng.normal(1000, 2, 300),
        rng.normal(1500, 2, 300),
        rng.normal(2000, 2, 300),
    ])

    cfg = {
        "calibration": {
            "peak_prominence": 5,
            "peak_width": 1,
            "nominal_adc": {"Po210": 1000, "Po218": 1500, "Po214": 2000},
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 2.0,
            "init_tau_adc": 0.0,
            "peak_search_radius": 5,
            "known_energies": {"Po210": 5.1, "Po214": 8.2},
            "sanity_tolerance_mev": 1.0,
        }
    }

    out = derive_calibration_constants(adc, cfg)

    a, _ = out["a"]
    c, _ = out["c"]

    assert pytest.approx(a * 1000 + c, rel=1e-3) == 5.1
    assert pytest.approx(a * 2000 + c, rel=1e-3) == 8.2


def test_calibration_sanity_check_triggers_error():
    """Misidentified peaks should cause calibrate_run to raise."""
    rng = np.random.default_rng(2)
    # True peaks: 800 (Po210), 1000 (Po218), 1200 (Po214)
    adc = np.concatenate([
        rng.normal(800, 2, 300),
        rng.normal(1000, 2, 300),
        rng.normal(1200, 2, 300),
    ])

    cfg = {
        "calibration": {
            "peak_prominence": 5,
            "peak_width": 1,
            # Swap Po210 and Po218 guesses to misidentify peaks
            "nominal_adc": {"Po210": 1000, "Po218": 800, "Po214": 1200},
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 4.0,
            "init_tau_adc": 0.0,
            "peak_search_radius": 5,
            "sanity_tolerance_mev": 0.5,
        }
    }

    with pytest.raises(RuntimeError):
        derive_calibration_constants(adc, cfg)


def test_calibrate_run_quadratic_option():
    """calibrate_run should support optional quadratic calibration."""
    rng = np.random.default_rng(3)
    adc = np.concatenate([
        rng.normal(1000, 2, 300),
        rng.normal(1500, 2, 300),
        rng.normal(2000, 2, 300),
    ])

    cfg = {
        "calibration": {
            "peak_prominence": 5,
            "peak_width": 1,
            "nominal_adc": {"Po210": 1000, "Po218": 1500, "Po214": 2000},
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 2.0,
            "init_tau_adc": 0.0,
            "peak_search_radius": 5,
            "quadratic": True,
            "sanity_tolerance_mev": 1.0,
        }
    }

    out = derive_calibration_constants(adc, cfg)

    a, _ = out["a"]
    a2, _ = out["a2"]
    c, _ = out["c"]

    adc_test = np.array([1000, 1500, 2000])
    energies = apply_calibration(adc_test, a, c, quadratic_coeff=a2)
    expected = np.array([
        DEFAULT_KNOWN_ENERGIES["Po210"],
        DEFAULT_KNOWN_ENERGIES["Po218"],
        DEFAULT_KNOWN_ENERGIES["Po214"],
    ])

    assert np.allclose(energies, expected, rtol=1e-3)
