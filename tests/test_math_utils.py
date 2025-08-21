import numpy as np
from math_utils import log_expm1_stable

def test_log_expm1_stable_matches_numpy():
    y = np.array([-50.0, -1e-6, 0.0, 1e-6, 1.0, 10.0])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        expected = np.log(np.expm1(y))
    tiny = np.log(np.finfo(float).tiny)
    expected = np.where(np.isfinite(expected), expected, tiny)
    result = log_expm1_stable(y)
    assert np.allclose(result, expected, rtol=1e-12, atol=0.0)

def test_log_expm1_stable_large_monotonic():
    y_large = np.array([800.0, 1000.0])
    res = log_expm1_stable(y_large)
    assert np.all(np.isfinite(res))
    assert res[1] > res[0]
