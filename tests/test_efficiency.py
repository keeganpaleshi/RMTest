import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efficiency import (
    calc_spike_efficiency,
    calc_assay_efficiency,
    calc_decay_efficiency,
    blue_combine,
)


def test_calc_spike_efficiency():
    eff = calc_spike_efficiency(50, 10.0, 100.0)
    assert eff == pytest.approx(0.05)


def test_calc_spike_efficiency_negative_counts():
    with pytest.raises(ValueError):
        calc_spike_efficiency(-1, 10.0, 100.0)


def test_calc_assay_efficiency():
    eff = calc_assay_efficiency(0.5, 2.0)
    assert eff == pytest.approx(0.25)


def test_calc_assay_efficiency_negative_rate():
    with pytest.raises(ValueError):
        calc_assay_efficiency(-0.1, 2.0)


def test_calc_decay_efficiency():
    eff = calc_decay_efficiency(0.9, 1.0)
    assert eff == pytest.approx(0.9)


def test_calc_decay_efficiency_negative_observed():
    with pytest.raises(ValueError):
        calc_decay_efficiency(-0.1, 1.0)


def test_blue_combine_uncorrelated():
    vals = np.array([0.4, 0.5, 0.6])
    errs = np.array([0.1, 0.2, 0.1])
    combined, sigma, weights = blue_combine(vals, errs)
    expected = np.average(vals, weights=1 / errs**2)
    expected_sigma = (1 / np.sum(1 / errs**2)) ** 0.5
    assert combined == pytest.approx(expected)
    assert sigma == pytest.approx(expected_sigma)
    assert len(weights) == 3


def test_blue_combine_correlated():
    vals = np.array([1.0, 2.0])
    errs = np.array([0.1, 0.2])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    combined, sigma, _ = blue_combine(vals, errs, corr)
    cov = corr * np.outer(errs, errs)
    Vinv = np.linalg.inv(cov)
    ones = np.ones(2)
    exp_sigma = (1 / (ones @ Vinv @ ones)) ** 0.5
    exp_val = (Vinv @ ones / (ones @ Vinv @ ones)) @ vals
    assert combined == pytest.approx(exp_val)
    assert sigma == pytest.approx(exp_sigma)


def test_blue_combine_negative_weights_warning():
    vals = np.array([1.0, 2.0])
    errs = np.array([0.1, 0.2])
    corr = np.array([[1.0, 0.99], [0.99, 1.0]])
    with pytest.warns(UserWarning):
        _, _, weights = blue_combine(vals, errs, corr)
    assert np.any(weights < 0)
