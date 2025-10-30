import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rmtest.external_rn_loader import load_external_rn_series


def timestamps(*values):
    return [pd.Timestamp(v, tz="UTC") for v in values]


def test_constant_mode_returns_constant_value():
    cfg = {"mode": "constant", "constant_bq_per_m3": 55.0}
    target = timestamps("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    result = load_external_rn_series(cfg, target)

    assert [ts for ts, _ in result] == target
    assert [val for _, val in result] == [55.0, 55.0]


def test_file_mode_nearest_interpolation(tmp_path):
    df = pd.DataFrame(
        {
            "time": ["2024-01-01T00:00:00", "2024-01-01T00:10:00"],
            "rn": [10.0, 20.0],
        }
    )
    path = tmp_path / "external.csv"
    df.to_csv(path, index=False)

    cfg = {
        "mode": "file",
        "file_path": str(path),
        "time_column": "time",
        "value_column": "rn",
        "interpolation": "nearest",
        "allowed_skew_seconds": 300,
    }

    target = timestamps("2024-01-01T00:02:00Z", "2024-01-01T00:09:00Z")

    result = load_external_rn_series(cfg, target)
    values = [val for _, val in result]

    assert values == [10.0, 20.0]


def test_file_mode_ffill_with_constant_fallback(tmp_path):
    df = pd.DataFrame(
        {
            "time": ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"],
            "rn": [5.0, 15.0],
        }
    )
    path = tmp_path / "ffill.csv"
    df.to_csv(path, index=False)

    cfg = {
        "mode": "file",
        "file_path": str(path),
        "time_column": "time",
        "value_column": "rn",
        "interpolation": "ffill",
        "allowed_skew_seconds": 600,
        "constant_bq_per_m3": 30.0,
    }

    target = timestamps(
        "2024-01-01T00:03:00Z",
        "2024-01-01T00:12:00Z",
        "2024-01-01T00:20:00Z",
    )

    result = load_external_rn_series(cfg, target)
    values = [val for _, val in result]

    assert values == [5.0, 15.0, 30.0]


def test_file_mode_applies_timezone(tmp_path):
    df = pd.DataFrame(
        {
            "time": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "rn": [10.0, 12.0],
        }
    )
    path = tmp_path / "tz.csv"
    df.to_csv(path, index=False)

    cfg = {
        "mode": "file",
        "file_path": str(path),
        "time_column": "time",
        "value_column": "rn",
        "timezone": "US/Eastern",
        "allowed_skew_seconds": 600,
    }

    target = timestamps("2024-01-01T05:00:00Z", "2024-01-01T06:00:00Z")

    result = load_external_rn_series(cfg, target)
    values = [val for _, val in result]

    assert values == [10.0, 12.0]


def test_missing_file_without_constant_raises(tmp_path):
    missing_path = tmp_path / "missing.csv"
    cfg = {
        "mode": "file",
        "file_path": str(missing_path),
        "default_bq_per_m3": None,
    }

    with pytest.raises(FileNotFoundError) as excinfo:
        load_external_rn_series(cfg, timestamps("2024-01-01T00:00:00Z"))

    assert "external radon file not found" in str(excinfo.value)
