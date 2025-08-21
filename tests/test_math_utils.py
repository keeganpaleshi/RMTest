import numpy as np

from math_utils import log_expm1_stable


def test_log_expm1_matches_numpy():
    y = np.array([-50, -1e-6, 0.0, 1e-6, 1.0, 10.0])
    expected = np.log(np.expm1(y))
    result = log_expm1_stable(y)
    assert np.allclose(result, expected, rtol=1e-12, atol=0.0)


def test_log_expm1_large_values():
    y = np.array([800.0, 1000.0])
    out = log_expm1_stable(y)
    assert np.isfinite(out).all()
    assert out[1] > out[0]

