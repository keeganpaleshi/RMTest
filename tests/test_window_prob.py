import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils


def test_window_prob_mixed_sigma():
    E = np.array([0.5, 1.0, 1.5, 2.0])
    sigma = np.array([0.1, 0.0, 0.2, 0.0])
    lo, hi = 0.9, 1.1
    probs = utils.window_prob(E, sigma, lo, hi)
    expected0 = norm.cdf(hi, loc=E[0], scale=sigma[0]) - norm.cdf(lo, loc=E[0], scale=sigma[0])
    expected2 = norm.cdf(hi, loc=E[2], scale=sigma[2]) - norm.cdf(lo, loc=E[2], scale=sigma[2])
    assert probs[0] == pytest.approx(expected0)
    assert probs[1] == pytest.approx(1.0)
    assert probs[2] == pytest.approx(expected2)
    assert probs[3] == pytest.approx(0.0)


def test_window_prob_scalar_zero_sigma():
    assert utils.window_prob(5.0, 0.0, 4.0, 6.0) == pytest.approx(1.0)
    assert utils.window_prob(7.0, 0.0, 4.0, 6.0) == pytest.approx(0.0)
