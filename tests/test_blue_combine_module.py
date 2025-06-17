import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blue_combine import blue_combine


def test_blue_combine_module_runs():
    vals = np.array([1.0, 2.0])
    errs = np.array([0.1, 0.2])
    combined, sigma, weights = blue_combine(vals, errs)
    expected = np.average(vals, weights=1 / errs**2)
    expected_sigma = (1 / np.sum(1 / errs**2)) ** 0.5
    assert combined == pytest.approx(expected)
    assert sigma == pytest.approx(expected_sigma)
    assert len(weights) == 2
