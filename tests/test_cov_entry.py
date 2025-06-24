import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from fitting import FitResult, FitParams


def test_cov_entry_valid_params():
    params = {
        "fit_valid": True,
        "dextra": 0.0,
        "A": 1.0,
        "B": 2.0,
        "C": 3.0,
    }
    cov = np.array(
        [
            [1.0, 0.1, 0.2],
            [0.1, 4.0, 0.3],
            [0.2, 0.3, 9.0],
        ]
    )
    index = {"A": 0, "B": 1, "C": 2}
    fr = FitResult(params, cov, 0, index)
    assert fr.get_cov("A", "B") == pytest.approx(0.1)
    assert analyze._cov_entry(fr, "C", "A") == pytest.approx(0.2)
    assert analyze._cov_entry(fr, "B", "C") == pytest.approx(0.3)


def test_cov_entry_missing_params():
    params = {"A": 1.0, "B": 2.0}
    cov = np.eye(2)
    fr = FitResult(params, cov, 0, {"A": 0, "B": 1})

    # One missing
    with pytest.raises(KeyError):
        analyze._cov_entry(fr, "A", "C")

    # Both missing should still raise
    with pytest.raises(KeyError):
        analyze._cov_entry(fr, "C", "D")

    fr_none = FitResult(params, None, 0, {"A": 0, "B": 1})
    assert analyze._cov_entry(fr_none, "A", "B") == 0.0
