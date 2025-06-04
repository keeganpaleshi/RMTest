import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import two_point_calibration, apply_calibration


def test_two_point_calibration():
    # Synthetic ADC peaks
    peak_adc = {"Po210": 1000, "Po214": 2000, "Po218": 1500}
    known = {"Po210": 5.30, "Po214": 7.69, "Po218": 6.00}
    m, c = two_point_calibration(
        [peak_adc["Po210"], peak_adc["Po214"]],
        [known["Po210"], known["Po214"]],
    )
    # Check that calibration maps 1000->5.30 and 2000->7.69
    assert pytest.approx(m * 1000 + c, rel=1e-3) == 5.30
    assert pytest.approx(m * 2000 + c, rel=1e-3) == 7.69


def test_apply_calibration():
    slope, intercept = 0.005, 0.02
    adc_vals = np.array([0, 100, 200])
    energies = apply_calibration(adc_vals, slope, intercept)
    assert np.allclose(energies, np.array([0.02, 0.52, 1.02]))
