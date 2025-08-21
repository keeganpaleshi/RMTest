import numpy as np
from math_utils import log_expm1_stable


def test_log_expm1_matches_numpy():
    y = np.array([-50.0, -1e-6, 0.0, 1e-6, 1.0, 10.0])
    tiny = np.finfo(float).tiny
    expected = np.where(
        y > 0,
        y + np.log1p(-np.exp(-y)),
        np.log(np.maximum(np.expm1(y), tiny)),
    )
    result = log_expm1_stable(y)
    assert np.allclose(result, expected, rtol=1e-12, atol=0.0)


def test_large_values_finite_and_monotonic():
    y = np.array([800.0, 1000.0])
    result = log_expm1_stable(y)
    assert np.all(np.isfinite(result))
    assert result[1] > result[0]
