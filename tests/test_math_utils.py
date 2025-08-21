import numpy as np
import sys
from pathlib import Path

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math_utils import log_expm1_stable


def test_log_expm1_stable_agrees_with_numpy():
    vals = np.array([1e-10, 1e-5, 1.0, 20.0, 700.0])
    expected = np.log(np.expm1(vals))
    out = log_expm1_stable(vals)
    assert np.allclose(out, expected)


def test_log_expm1_stable_monotonic_large_inputs():
    a = log_expm1_stable(800.0)
    b = log_expm1_stable(1000.0)
    assert np.isfinite(a) and np.isfinite(b)
    assert b > a
