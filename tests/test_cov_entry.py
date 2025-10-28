import sys
from pathlib import Path
import math
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
from fitting import FitResult, FitParams
import pandas as pd



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
    fr = FitResult(params, cov, 0)
    assert fr.get_cov("A", "B") == pytest.approx(0.1)
    assert fr.get_cov("C", "A") == pytest.approx(0.2)
    assert fr.get_cov("B", "C") == pytest.approx(0.3)


def test_cov_entry_missing_params():
    params = {"A": 1.0, "B": 2.0}
    cov = np.eye(2)
    fr = FitResult(params, cov, 0)

    # Missing entries now return NaN rather than raising
    assert math.isnan(fr.get_cov("A", "C"))
    assert math.isnan(fr.get_cov("C", "D"))

    fr_none = FitResult(params, None, 0)
    assert math.isnan(fr_none.get_cov("A", "B"))


def test_cov_entry_prefix_handled():
    params = {
        "A": 1.0,
        "B": 2.0,
        "cov_A_B": 0.1,
    }
    cov = np.array([[1.0, 0.1], [0.1, 4.0]])
    fr = FitResult(params, cov, 0)
    assert len(fr.param_index) == 2
    assert "cov_A_B" not in fr.param_index
    assert fr.get_cov("A", "B") == pytest.approx(0.1)


def test_fit_result_cov_mismatch():
    params = {"A": 1.0, "B": 2.0, "C": 3.0}
    cov = np.ones((2, 3))
    with pytest.raises(ValueError):
        FitResult(params, cov, 0)


def test_fit_result_cov_ok():
    params = {"A": 1.0, "B": 2.0, "C": 3.0}
    cov = np.eye(3)
    FitResult(params, cov, 0)


def test_cov_dataframe_created():
    params = {"A": 1.0, "B": 2.0}
    cov = np.array([[1.0, 0.1], [0.1, 4.0]])
    fr = FitResult(params, cov, 0)
    assert isinstance(fr.cov_df, pd.DataFrame)
    assert fr.cov_df.loc["A", "B"] == pytest.approx(0.1)
