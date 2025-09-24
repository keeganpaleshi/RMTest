import numpy as np
from math_utils import log_expm1_stable


def test_log_expm1_stable_matches_numpy():
    y = np.array([-50.0, -1e-6, 0.0, 1e-6, 1.0, 10.0])
    result = log_expm1_stable(y)

    expected = np.log(np.expm1(y))
    pos = y > 0
    expected[pos] = y[pos] + np.log1p(-np.exp(-y[pos]))

    np.testing.assert_allclose(result[pos], expected[pos], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(result[~pos], expected[~pos])


def test_log_expm1_stable_negative_inputs_propagate_nan():
    res = log_expm1_stable(np.array([-1.0, -2.5]))
    assert np.isnan(res[0])
    assert np.isnan(res[1])


def test_log_expm1_stable_large_monotonic():
    vals = log_expm1_stable(np.array([800.0, 1000.0]))
    assert np.isfinite(vals[0]) and np.isfinite(vals[1])
    assert vals[1] > vals[0]
