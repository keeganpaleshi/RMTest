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
    derive_calibration_constants,
)
import calibration as calib_mod
import constants
import emg_stable
from constants import DEFAULT_KNOWN_ENERGIES


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


def test_configure_emg_updates_tau_floor():
    """configure_emg should sync tau floors across modules and fallbacks."""

    original_tau = calib_mod.get_emg_tau_min()
    original_use_stable = calib_mod.get_use_stable_emg()

    tau_floor = 5e-3
    calib_mod.configure_emg(True, tau_floor)
    try:
        assert constants._TAU_MIN == pytest.approx(tau_floor)

        x = np.linspace(-1.0, 1.0, 5)
        gaussian_pdf = gaussian(x, 0.0, 1.0)
        stable_pdf = emg_stable.StableEMG().pdf(
            x, mu=0.0, sigma=1.0, tau=tau_floor / 2, amplitude=1.0
        )

        assert np.allclose(stable_pdf, gaussian_pdf)
    finally:
        calib_mod.configure_emg(original_use_stable, original_tau)


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
    assert set(out.peaks.keys()) == {"Po210", "Po218", "Po214"}

    cfg_bad = {"calibration": dict(base_cfg["calibration"])}
    cfg_bad["calibration"]["peak_search_radius"] = 1

    with pytest.raises(RuntimeError):
        derive_calibration_constants(adc, cfg_bad)


def test_calibration_uses_known_energies_from_config():
    """Custom energies in config should override defaults."""
    rng = np.random.default_rng(1)
    adc = np.concatenate(
        [
            rng.normal(1000, 2, 300),
            rng.normal(1500, 2, 300),
            rng.normal(2000, 2, 300),
        ]
    )

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

    res = derive_calibration_constants(adc, cfg)

    assert pytest.approx(res.predict(1000), rel=1e-3) == 5.1
    assert pytest.approx(res.predict(2000), rel=1e-3) == 8.2


def test_calibration_sanity_check_triggers_error():
    """Misidentified peaks should cause calibrate_run to raise."""
    rng = np.random.default_rng(2)
    # True peaks: 800 (Po210), 1000 (Po218), 1200 (Po214)
    adc = np.concatenate(
        [
            rng.normal(800, 2, 300),
            rng.normal(1000, 2, 300),
            rng.normal(1200, 2, 300),
        ]
    )

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


def test_peak_ordering_violation_raises():
    """Swapped nominal guesses should trigger ordering check."""
    rng = np.random.default_rng(42)
    adc = np.concatenate(
        [
            rng.normal(800, 2, 300),
            rng.normal(1000, 2, 300),
            rng.normal(1200, 2, 300),
        ]
    )

    cfg = {
        "calibration": {
            "peak_prominence": 5,
            "peak_width": 1,
            # Swap Po214 and Po218 guesses so Po218 peak lies above Po214
            "nominal_adc": {"Po210": 800, "Po218": 1200, "Po214": 1000},
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 4.0,
            "init_tau_adc": 0.0,
            "peak_search_radius": 5,
            # Large tolerance to bypass energy sanity check
            "sanity_tolerance_mev": 10.0,
        }
    }

    with pytest.raises(RuntimeError, match="inconsistent peak ordering"):
        derive_calibration_constants(adc, cfg)


def test_calibrate_run_quadratic_option(caplog):
    """calibrate_run should warn and fall back to linear when quadratic requested."""
    rng = np.random.default_rng(3)
    adc = np.concatenate(
        [
            rng.normal(1000, 2, 300),
            rng.normal(1500, 2, 300),
            rng.normal(2000, 2, 300),
        ]
    )

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

    adc_test = np.array([1000, 1500, 2000])
    energies = out.predict(adc_test)
    assert np.allclose(
        energies,
        [
            DEFAULT_KNOWN_ENERGIES["Po210"],
            DEFAULT_KNOWN_ENERGIES["Po218"],
            DEFAULT_KNOWN_ENERGIES["Po214"],
        ],
        rtol=1e-3,
    )
    assert out.coeffs[2] != 0.0


def test_use_quadratic_auto():
    rng = np.random.default_rng(4)
    adc = np.concatenate(
        [
            rng.normal(1000, 2, 300),
            rng.normal(1500, 2, 300),
            rng.normal(2000, 2, 300),
        ]
    )

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
            "use_quadratic": "auto",
            "sanity_tolerance_mev": 1.0,
        }
    }

    out = derive_calibration_constants(adc, cfg)

    assert len(out.coeffs) == 3


def test_energy_uncertainty_clipping():
    """Negative covariance should not produce NaNs in uncertainty."""
    import pandas as pd
    from calibration import CalibrationResult

    events = pd.DataFrame({"adc": [1.0]})
    a_sig = 0.001
    a2_sig = 0.002
    c_sig = 0.02
    cov_ac = -0.1
    cov_a_a2 = -0.05
    cov_a2_c = -0.02

    var_energy = (
        (events["adc"] * a_sig) ** 2
        + (events["adc"] ** 2 * a2_sig) ** 2
        + c_sig**2
        + 2 * events["adc"] * cov_ac
        + 2 * events["adc"] ** 3 * cov_a_a2
        + 2 * events["adc"] ** 2 * cov_a2_c
    )
    assert var_energy.iloc[0] < 0

    cov = np.array(
        [
            [c_sig**2, cov_ac, cov_a2_c],
            [cov_ac, a_sig**2, cov_a_a2],
            [cov_a2_c, cov_a_a2, a2_sig**2],
        ]
    )
    calib = CalibrationResult(coeffs=[0.0, 1.0, 0.0], covariance=cov)

    expected = np.sqrt(np.clip(var_energy, 0, None))
    events["denergy_MeV"] = calib.uncertainty(events["adc"])

    assert np.allclose(events["denergy_MeV"], expected)
    assert np.isfinite(events["denergy_MeV"]).all()


def test_calibrationresult_uncertainty_linear():
    """CalibrationResult.uncertainty should match analytic propagation."""
    from calibration import CalibrationResult

    cov = np.array([[0.2**2, 0.0], [0.0, 0.1**2]])
    calib = CalibrationResult(coeffs=[1.0, 2.0], cov=cov)

    adc = np.array([5.0])
    expected = np.sqrt((adc * 0.1) ** 2 + 0.2**2)

    assert np.allclose(calib.uncertainty(adc), expected)


def test_calibrationresult_uncertainty_quadratic():
    """Quadratic coefficient and covariance should propagate correctly."""
    from calibration import CalibrationResult

    cov = np.array(
        [
            [0.1**2, 0.0, 0.0],
            [0.0, 0.1**2, 0.005],
            [0.0, 0.005, 0.02**2],
        ]
    )
    calib = CalibrationResult(coeffs=[0.5, 1.0, 0.05], cov=cov)

    adc = 2.0
    var = (
        (adc * 0.1) ** 2
        + (adc**2 * 0.02) ** 2
        + 0.1**2
        + 2 * adc * 0.0
        + 2 * adc**3 * 0.005
    )
    expected = np.sqrt(var)

    assert np.allclose(calib.uncertainty(adc), expected)


def test_calibrationresult_uncertainty_negative_covariance():
    """Non-positive covariance should not yield NaN uncertainties."""
    from calibration import CalibrationResult

    cov = np.array([[0.02**2, -0.5], [-0.5, 0.001**2]])
    calib = CalibrationResult(coeffs=[0.0, 1.0], cov=cov)

    sigma = calib.uncertainty([1.0])
    assert np.isfinite(sigma).all()

def test_intercept_fit_two_point_deprecation(monkeypatch):
    import calibration
    import calibrate

    called = {}

    def stub(*args, **kwargs):
        called["called"] = True

    monkeypatch.setattr(calibrate, "intercept_fit_two_point", stub)

    with pytest.deprecated_call():
        calibration.intercept_fit_two_point(1, 2)

    assert called["called"]


def test_two_point_fallback_to_one_point_warns():
    rng = np.random.default_rng(123)
    adc = rng.normal(1800, 2, 500)
    cfg = {
        "calibration": {
            "slope_MeV_per_ch": 0.00435,
            "float_slope": False,
            "use_two_point": True,
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

    with pytest.warns(RuntimeWarning, match="Two-point calibration failed"):
        res = derive_calibration_constants(adc, cfg)

    assert res.coeffs[1] == 0.00435
    expected_c = 7.687 - 0.00435 * 1800
    assert res.coeffs[0] == pytest.approx(expected_c, abs=0.02)


def _synth_emg_calibration_cfg():
    return {
        "calibration": {
            "peak_prominence": 5,
            "peak_width": 1,
            "nominal_adc": {"Po210": 1000, "Po218": 1500, "Po214": 2000},
            "fit_window_adc": 25,
            "use_emg": True,
            "init_sigma_adc": 3.0,
            "init_tau_adc": 0.0,
            "peak_search_radius": 10,
            "sanity_tolerance_mev": 2.0,
        }
    }


def _synth_adc_sample(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(1000, 3, 400),
            rng.normal(1500, 3, 400),
            rng.normal(2000, 3, 400),
        ]
    )


def test_calibration_uses_configured_curve_fit_evals(monkeypatch):
    adc = _synth_adc_sample(12)
    cfg = _synth_emg_calibration_cfg()
    cfg["calibration"]["fit_maxfev"] = 1234
    cfg["calibration"]["curve_fit_max_evaluations"] = 9999

    recorded = []

    def fake_curve_fit(func, xdata, ydata, p0=None, bounds=(-np.inf, np.inf), **kwargs):
        recorded.append(kwargs.get("maxfev"))
        return np.asarray(p0), np.eye(len(p0))

    monkeypatch.setattr(calib_mod, "curve_fit", fake_curve_fit)

    derive_calibration_constants(adc, cfg)

    assert recorded  # ensure the fake was exercised
    assert all(val == 1234 for val in recorded)


def test_calibration_tau_bounds_defaults(monkeypatch):
    adc = _synth_adc_sample(10)
    cfg = _synth_emg_calibration_cfg()

    captured_bounds = []

    def fake_curve_fit(func, xdata, ydata, p0=None, bounds=(-np.inf, np.inf), **kwargs):
        captured_bounds.append(bounds)
        return np.asarray(p0), np.eye(len(p0))

    monkeypatch.setattr(calib_mod, "curve_fit", fake_curve_fit)

    res = derive_calibration_constants(adc, cfg)
    assert set(res.peaks) == {"Po210", "Po218", "Po214"}

    emg_bounds = [b for b in captured_bounds if len(b[0]) == 4]
    assert len(emg_bounds) == 2

    po210_bounds, po218_bounds = emg_bounds
    assert po210_bounds[1][3] == pytest.approx(50.0)
    assert po218_bounds[1][3] == pytest.approx(8.0)


def test_calibration_tau_bounds_overrides(monkeypatch):
    adc = _synth_adc_sample(11)
    cfg = _synth_emg_calibration_cfg()
    cfg["calibration"]["tau_bounds_adc"] = {
        "default": (0.002, 9.5),
        "Po218": (0.002, 3.5),
    }

    captured_bounds = []

    def fake_curve_fit(func, xdata, ydata, p0=None, bounds=(-np.inf, np.inf), **kwargs):
        captured_bounds.append(bounds)
        return np.asarray(p0), np.eye(len(p0))

    monkeypatch.setattr(calib_mod, "curve_fit", fake_curve_fit)

    res = derive_calibration_constants(adc, cfg)
    assert set(res.peaks) == {"Po210", "Po218", "Po214"}

    emg_bounds = [b for b in captured_bounds if len(b[0]) == 4]
    assert len(emg_bounds) == 2

    po210_bounds, po218_bounds = emg_bounds
    assert po210_bounds[0][3] == pytest.approx(0.002)
    assert po210_bounds[1][3] == pytest.approx(9.5)
    assert po218_bounds[0][3] == pytest.approx(0.002)
    assert po218_bounds[1][3] == pytest.approx(3.5)
