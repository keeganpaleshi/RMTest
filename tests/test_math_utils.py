import numpy as np
import pytest

from math_utils import log_expm1_stable


def test_log_expm1_stable_matches_numpy():
    y = np.array([-50.0, -1e-6, 0.0, 1e-6, 1.0, 10.0])
    result = log_expm1_stable(y)
    tiny = np.finfo(float).tiny
    expected = np.empty_like(y)
    mask = y > 0
    expected[mask] = y[mask] + np.log1p(-np.exp(-y[mask]))
    expm1_vals = np.expm1(y[~mask])
    y_clamped = np.maximum(expm1_vals, tiny)
    expected[~mask] = np.log(np.expm1(y_clamped))
    assert np.allclose(result, expected, rtol=1e-12)


def test_log_expm1_stable_large_values_monotonic_and_finite():
    y = np.array([800.0, 1000.0])
    result = log_expm1_stable(y)
    assert np.all(np.isfinite(result))
    # Ensure monotonic increasing
    assert np.all(np.diff(result) > 0)
