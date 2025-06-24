import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import CalibrationResult


def test_get_cov_by_name_and_exponent():
    cov = np.array([
        [1.0, 0.1, 0.2],
        [0.1, 4.0, 0.3],
        [0.2, 0.3, 9.0],
    ])
    calib = CalibrationResult(coeffs=[0.0, 1.0, 2.0], cov=cov)
    assert calib.get_cov("c", "a") == pytest.approx(0.1)
    assert calib.get_cov(2, "c") == pytest.approx(0.2)
    assert calib.get_cov("a", 2) == pytest.approx(0.3)


def test_get_cov_missing():
    cov = np.eye(2)
    calib = CalibrationResult(coeffs=[1.0, 2.0], cov=cov)
    with pytest.raises(KeyError):
        calib.get_cov("a2", "a")

