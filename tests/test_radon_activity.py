import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radon_activity import compute_radon_activity, compute_total_radon


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


def test_compute_radon_activity_only_218_error():
    a, s = compute_radon_activity(10.0, 1.0, 1.0, 12.0, None, 1.0)
    assert a == pytest.approx(10.0)
    assert s == pytest.approx(1.0)


def test_compute_total_radon():
    conc, dconc, tot, dtot = compute_total_radon(5.0, 0.5, 10.0, 20.0)
    assert conc == pytest.approx(0.5)
    assert dconc == pytest.approx(0.05)
    assert tot == pytest.approx(10.0)
    assert dtot == pytest.approx(1.0)
