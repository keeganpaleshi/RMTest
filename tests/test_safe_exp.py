import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import _safe_exp, EXP_OVERFLOW_DOUBLE


def test_safe_exp_clipping():
    values = np.array([-2 * EXP_OVERFLOW_DOUBLE, 0.0, 2 * EXP_OVERFLOW_DOUBLE])
    out = _safe_exp(values)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(np.exp(-EXP_OVERFLOW_DOUBLE))
    assert out[-1] == pytest.approx(np.exp(EXP_OVERFLOW_DOUBLE))
