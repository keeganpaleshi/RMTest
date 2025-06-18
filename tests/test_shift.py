import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from systematics import apply_linear_adc_shift


def test_larger_slope_gives_larger_shift():
    adc = np.array([0.0, 1.0, 2.0])
    t = np.array([0.0, 1.0, 2.0])
    small = apply_linear_adc_shift(adc, t, 1.0)
    large = apply_linear_adc_shift(adc, t, 2.0)
    assert np.all(large >= small)
