import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radon_activity import (
    compute_radon_activity,
    compute_total_radon,
    radon_activity_curve,
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


def test_compute_radon_activity_only_214_error():
    a, s = compute_radon_activity(10.0, None, 1.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(2.0)


def test_compute_radon_activity_only_214_error_eff_not_one():
    a, s = compute_radon_activity(10.0, None, 0.8, 12.0, 2.0, 0.9)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(2.0)


def test_compute_radon_activity_mixed_efficiency():
    a, s = compute_radon_activity(10.0, 1.0, 0.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(2.0)


def test_compute_radon_activity_only_218_error():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, None, 1.0)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_only_218_error_eff_not_one():
    a, s = compute_radon_activity(10.0, 1.0, 0.7, 12.0, None, 0.6)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_only_214_error_zero_218_error():
    a, s = compute_radon_activity(10.0, 0.0, 1.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(2.0)


def test_compute_radon_activity_only_218_error_zero_214_error():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, 0.0, 1.0)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(1.0)


def test_compute_radon_activity_mixed_error_sign():
    a, s = compute_radon_activity(10.0, -1.0, 1.0, 12.0, 2.0, 1.0)
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(2.0)


def test_compute_radon_activity_mixed_error_sign_214():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, -2.0, 1.0)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(1.0)


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


def test_compute_total_radon():
    conc, dconc, tot, dtot = compute_total_radon(5.0, 0.5, 10.0, 20.0)
    assert conc == pytest.approx(0.5)
    assert dconc == pytest.approx(0.05)
    assert tot == pytest.approx(10.0)
    assert dtot == pytest.approx(1.0)


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
