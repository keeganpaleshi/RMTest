import numpy as np
from math_utils import log_expm1_stable


def test_log_expm1_stable_matches_numpy():
    y = np.array([-50.0, -1e-6, 0.0, 1e-6, 1.0, 10.0])
    expected = np.empty_like(y)
    mask = y > 0
    expected[mask] = y[mask] + np.log1p(-np.exp(-y[mask]))
    expm1_vals = np.expm1(y[~mask])
    tiny = np.finfo(float).tiny
    expm1_vals = np.clip(expm1_vals, tiny, None)
    expected[~mask] = np.log(expm1_vals)
    actual = log_expm1_stable(y)
    assert np.allclose(actual, expected, rtol=1e-12, atol=0)


def test_log_expm1_stable_large_values():
    vals = np.array([800.0, 1000.0])
    out = log_expm1_stable(vals)
    assert np.isfinite(out).all()
    assert out[1] > out[0]
