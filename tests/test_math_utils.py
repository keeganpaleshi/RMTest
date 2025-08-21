import numpy as np
from math_utils import log_expm1_stable


def test_log_expm1_stable_matches_numpy():
    y = np.array([-50.0, -1e-6, 0.0, 1e-6, 1.0, 10.0])
    tiny = np.finfo(float).tiny
    expected = np.empty_like(y)
    mask = y > 0
    expected[mask] = y[mask] + np.log1p(-np.exp(-y[mask]))
    tmp = np.clip(np.expm1(y[~mask]), tiny, None)
    expected[~mask] = np.log(tmp)
    np.testing.assert_allclose(
        log_expm1_stable(y), expected, rtol=1e-12, atol=0
    )


def test_log_expm1_stable_large_monotonic():
    vals = log_expm1_stable(np.array([800.0, 1000.0]))
    assert np.isfinite(vals[0]) and np.isfinite(vals[1])
    assert vals[1] > vals[0]
