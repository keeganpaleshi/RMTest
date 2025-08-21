import numpy as np
from numpy.testing import assert_allclose

from math_utils import log_expm1_stable


def test_log_expm1_stable_matches_numpy():
    y = np.array([-50.0, -1e-6, 0.0, 1e-6, 1.0, 10.0])
    res = log_expm1_stable(y)
    tiny = np.finfo(float).tiny
    expected = np.empty_like(y)
    pos = y > 0
    expected[pos] = np.log(np.expm1(y[pos]))
    expm1_val = np.expm1(y[~pos])
    expected[~pos] = np.log(np.clip(expm1_val, tiny, None))
    assert np.all(np.isfinite(res))
    assert_allclose(res, expected, rtol=1e-12)


def test_log_expm1_stable_large_monotonic():
    y = np.array([800.0, 1000.0])
    res = log_expm1_stable(y)
    assert np.all(np.isfinite(res))
    assert res[1] > res[0]
