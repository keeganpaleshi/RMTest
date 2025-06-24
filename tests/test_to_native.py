import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import to_native


def test_to_native_numpy_scalar_and_array():
    assert to_native(np.int64(5)) == 5
    assert isinstance(to_native(np.int64(5)), int)
    arr = np.array([1, 2, 3], dtype=np.float32)
    assert to_native(arr) == [1.0, 2.0, 3.0]


def test_to_native_pandas_objects():
    ts = pd.Timestamp("2023-01-01T00:00:00Z")
    td = pd.Timedelta(seconds=30)
    assert to_native(ts) == ts.isoformat()
    assert to_native(td) == td.isoformat()
    assert to_native(pd.NA) is None


def test_to_native_datetime_object():
    dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert to_native(dt) == dt.isoformat()


@dataclass
class Point:
    x: int
    y: float
    data: np.ndarray


def test_to_native_dataclass():
    p = Point(1, 2.5, np.array([4, 5]))
    assert to_native(p) == {"x": 1, "y": 2.5, "data": [4, 5]}
