from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blue_combine import blue_combine, Measurements


def test_blue_combine_wrapper_import():
    m = Measurements(values=[1.0, 2.0], errors=[0.1, 0.2])
    result = blue_combine(m.values, m.errors)
    assert isinstance(result, tuple) and len(result) == 3

