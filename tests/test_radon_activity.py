import sys
from pathlib import Path
import logging
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radon_activity import (
    clamp_non_negative,
    compute_radon_activity,
    compute_total_radon,
    radon_activity_curve,
    radon_delta,
    print_activity_breakdown,
)
import math
import numpy as np


def test_compute_radon_activity_weighted():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, 2.0, 1.0)
    w1 = 1 / 1.0**2
    w2 = 1 / 2.0**2
    expected = (10.0 * w1 + 12.0 * w2) / (w1 + w2)
    err = (1 / (w1 + w2)) ** 0.5
    assert a == pytest.approx(expected)
    assert s == pytest.approx(err)


def test_compute_radon_activity_efficiencies_not_weighted():
    """Non-zero efficiencies should not scale the rates."""
    a_ref, s_ref = compute_radon_activity(10.0, 1.0, 1.0, 12.0, 2.0, 1.0)
    a, s = compute_radon_activity(10.0, 1.0, 0.6, 12.0, 2.0, 0.7)
    assert a == pytest.approx(a_ref)
    assert s == pytest.approx(s_ref)


def test_compute_radon_activity_only_214_error():
    a, s = compute_radon_activity(10.0, None, 1.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx((10.0 + 12.0) / 2)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_only_214_error_eff_not_one():
    a, s = compute_radon_activity(10.0, None, 0.8, 12.0, 2.0, 0.9)
    assert a == pytest.approx((10.0 + 12.0) / 2)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_mixed_efficiency():
    a, s = compute_radon_activity(10.0, 1.0, 0.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(2.0)


def test_compute_radon_activity_only_218_error():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, None, 1.0)
    assert a == pytest.approx((10.0 + 12.0) / 2)
    assert s == pytest.approx(0.5)


def test_compute_radon_activity_only_218_error_eff_not_one():
    a, s = compute_radon_activity(10.0, 1.0, 0.7, 12.0, None, 0.6)
    assert a == pytest.approx((10.0 + 12.0) / 2)
    assert s == pytest.approx(0.5)


def test_compute_radon_activity_only_214_error_zero_218_error():
    a, s = compute_radon_activity(10.0, 0.0, 1.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(0.0)


def test_compute_radon_activity_only_218_error_zero_214_error():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, 0.0, 1.0)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(0.0)


def test_compute_radon_activity_mixed_error_sign():
    a, s = compute_radon_activity(10.0, -1.0, 1.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx((10.0 + 12.0) / 2)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_mixed_error_sign_214():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, -2.0, 1.0)
    assert a == pytest.approx((10.0 + 12.0) / 2)
    assert s == pytest.approx(0.5)


def test_compute_radon_activity_mixed_efficiency_214():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, 2.0, 0.0)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_single_214():
    a, s = compute_radon_activity(None, None, 1.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(2.0)


def test_compute_radon_activity_single_218():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, None, None, 1.0)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_single_214_eff_not_one():
    """Efficiency values should not scale single-isotope rates."""
    a, s = compute_radon_activity(None, None, 0.5, 12.0, 2.0, 0.7)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(2.0)


def test_compute_radon_activity_single_218_eff_not_one():
    """Efficiency values should not scale single-isotope rates."""
    a, s = compute_radon_activity(10.0, 1.0, 0.3, None, None, 0.8)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_uncertainty_po214_only():
    """Returns unweighted average when only Po-214 error is valid."""
    a, s = compute_radon_activity(4.0, None, 1.0, 5.0, 0.3, 1.0)
    assert a == pytest.approx((4.0 + 5.0) / 2)
    assert s == pytest.approx(0.15)


def test_compute_radon_activity_negative_eff218():
    with pytest.raises(ValueError):
        compute_radon_activity(10.0, 1.0, -0.2, 12.0, 2.0, 1.0)


def test_compute_radon_activity_negative_eff214():
    with pytest.raises(ValueError):
        compute_radon_activity(10.0, 1.0, 1.0, 12.0, 2.0, -0.5)


def test_compute_radon_activity_equilibrium_check():
    """Fail when times are before secular equilibrium if required."""
    with pytest.raises(ValueError):
        compute_radon_activity(
            10.0,
            1.0,
            1.0,
            12.0,
            2.0,
            1.0,
            t_since_start=1.0,
            settle_time=10.0,
            require_equilibrium=True,
        )


def test_compute_total_radon():
    conc, dconc, tot, dtot = compute_total_radon(5.0, 0.5, 10.0, 20.0)
    assert conc == pytest.approx(0.5)
    assert dconc == pytest.approx(0.05)
    assert tot == pytest.approx(10.0)
    assert dtot == pytest.approx(1.0)


def test_compute_radon_activity_missing_uncertainty_returns_nan():
    """Single rate without uncertainty should propagate NaN error."""
    a, s = compute_radon_activity(5.0, None, 1.0, None, None, 1.0)
    assert a == pytest.approx(5.0)
    assert math.isnan(s)


def test_compute_radon_activity_unweighted_one_error_missing():
    """Average both rates when one uncertainty is missing."""
    a, s = compute_radon_activity(5.0, 0.5, 1.0, 7.0, None, 1.0)
    assert a == pytest.approx(6.0)
    assert s == pytest.approx(0.25)


def test_compute_radon_activity_both_missing_errors():
    """Average both rates with NaN uncertainty when no errors are given."""
    a, s = compute_radon_activity(5.0, None, 1.0, 7.0, None, 1.0)
    assert a == pytest.approx(6.0)
    assert math.isnan(s)


def test_compute_total_radon_negative_sample_volume():
    with pytest.raises(ValueError):
        compute_total_radon(5.0, 0.5, 10.0, -1.0)


def test_compute_total_radon_negative_err_bq():
    with pytest.raises(ValueError):
        compute_total_radon(5.0, -0.1, 10.0, 1.0)


def test_clamp_non_negative():
    val, err = clamp_non_negative(-0.02, 0.01)
    assert val == pytest.approx(0.0)
    assert err == pytest.approx(0.01)


def test_compute_total_radon_negative_activity_default_raises():
    with pytest.raises(RuntimeError):
        compute_total_radon(-1.0, 0.5, 10.0, 1.0)


def test_compute_total_radon_negative_activity_allowed(caplog):
    conc, dconc, tot, dtot = compute_total_radon(
        -1.0, 0.5, 10.0, 1.0, allow_negative_activity=True
    )
    assert conc == pytest.approx(-0.1)
    assert dconc == pytest.approx(0.05)
    assert tot == pytest.approx(-0.1)
    assert dtot == pytest.approx(0.05)


def test_radon_activity_curve():
    times = [0.0, 1.0]
    E = 5.0
    dE = 0.5
    N0 = 2.0
    dN0 = 0.2
    hl = 10.0
    act, err = radon_activity_curve(times, E, dE, N0, dN0, hl)
    lam = math.log(2.0) / hl
    import numpy as np

    exp_term = np.exp(-lam * np.asarray(times))
    expected = E * (1 - exp_term) + lam * N0 * exp_term
    var = ((1 - exp_term) * dE) ** 2 + ((lam * exp_term) * dN0) ** 2
    assert np.allclose(act, expected)
    assert np.allclose(err, np.sqrt(var))


def test_radon_activity_curve_with_covariance():
    times = [0.0, 1.0]
    E = 5.0
    dE = 0.5
    N0 = 2.0
    dN0 = 0.2
    hl = 10.0
    cov = 0.05
    act, err = radon_activity_curve(times, E, dE, N0, dN0, hl, cov)
    lam = math.log(2.0) / hl
    exp_term = np.exp(-lam * np.asarray(times))
    expected = E * (1 - exp_term) + lam * N0 * exp_term
    dA_dE = 1 - exp_term
    dA_dN0 = lam * exp_term
    var = (dA_dE * dE) ** 2 + (dA_dN0 * dN0) ** 2 + 2 * dA_dE * dA_dN0 * cov
    assert np.allclose(act, expected)
    assert np.allclose(err, np.sqrt(var))


def test_radon_delta():
    start = 0.0
    end = 2.0
    E = 5.0
    dE = 0.5
    N0 = 2.0
    dN0 = 0.2
    hl = 10.0
    delta, sigma = radon_delta(start, end, E, dE, N0, dN0, hl)

    lam = math.log(2.0) / hl
    exp1 = math.exp(-lam * start)
    exp2 = math.exp(-lam * end)
    expected = E * (exp1 - exp2) + lam * N0 * (exp2 - exp1)
    var = ((exp1 - exp2) * dE) ** 2 + ((lam * (exp2 - exp1)) * dN0) ** 2
    assert delta == pytest.approx(expected)
    assert sigma == pytest.approx(math.sqrt(var))


def test_radon_delta_with_covariance():
    start = 0.0
    end = 2.0
    E = 5.0
    dE = 0.5
    N0 = 2.0
    dN0 = 0.2
    hl = 10.0
    cov = 0.03
    delta, sigma = radon_delta(start, end, E, dE, N0, dN0, hl, cov)
    lam = math.log(2.0) / hl
    exp1 = math.exp(-lam * start)
    exp2 = math.exp(-lam * end)
    expected = E * (exp1 - exp2) + lam * N0 * (exp2 - exp1)
    d_delta_dE = exp1 - exp2
    d_delta_dN0 = lam * (exp2 - exp1)
    var = (
        (d_delta_dE * dE) ** 2
        + (d_delta_dN0 * dN0) ** 2
        + 2 * d_delta_dE * d_delta_dN0 * cov
    )
    assert delta == pytest.approx(expected)
    assert sigma == pytest.approx(math.sqrt(var))


def test_radon_activity_curve_invalid_half_life():
    with pytest.raises(ValueError):
        radon_activity_curve([0.0, 1.0], 1.0, 0.1, 2.0, 0.2, 0.0)
    with pytest.raises(ValueError):
        radon_activity_curve([0.0, 1.0], 1.0, 0.1, 2.0, 0.2, -5.0)


def test_radon_delta_invalid_half_life():
    with pytest.raises(ValueError):
        radon_delta(0.0, 2.0, 1.0, 0.1, 2.0, 0.2, 0.0)
    with pytest.raises(ValueError):
        radon_delta(0.0, 2.0, 1.0, 0.1, 2.0, 0.2, -5.0)


def test_print_activity_breakdown(capsys):
    rows = [
        {
            "iso": "Po218",
            "raw_rate": 0.112,
            "baseline_rate": 0.027,
            "corrected": 0.085,
            "err_raw": 0.012,
            "err_corrected": 0.01,
        },
        {
            "iso": "Po214",
            "raw_rate": 0.118,
            "baseline_rate": 0.026,
            "corrected": 0.092,
            "err_raw": 0.013,
            "err_corrected": 0.011,
        },
    ]

    print_activity_breakdown(rows)
    captured = capsys.readouterr().out
    assert "Total radon" in captured
