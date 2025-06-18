import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blue_combine import blue_combine


def test_blue_weights_normalization():
    vals = np.array([1.0, 2.0, 3.0])
    errs = np.array([0.1, 0.2, 0.3])
    _, _, weights = blue_combine(vals, errs)
    assert abs(np.sum(weights) - 1) < 1e-9
