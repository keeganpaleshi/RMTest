from __future__ import annotations

import pandas as pd
import pytest

from radon.external_rn_loader import load_external_rn_series


def _timestamps(count: int) -> list[pd.Timestamp]:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    return [start + pd.Timedelta(hours=i) for i in range(count)]


def test_constant_mode_uses_configured_value():
    cfg = {"mode": "constant", "constant_bq_per_m3": 42.5}
    targets = _timestamps(3)

    result = load_external_rn_series(cfg, targets)

    assert [val for _, val in result] == [42.5, 42.5, 42.5]
    assert all(ts.tzinfo is not None for ts, _ in result)


def test_constant_mode_defaults_to_80_when_missing_config():
    targets = _timestamps(2)

    result = load_external_rn_series(None, targets)

    assert [val for _, val in result] == [80.0, 80.0]


def test_file_mode_nearest_interpolation(tmp_path):
    csv_path = tmp_path / "external.csv"
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:10:00Z",
                "2024-01-01T00:20:00Z",
            ],
            "rn_bq_per_m3": [60.0, 90.0, 120.0],
        }
    )
    df.to_csv(csv_path, index=False)

    cfg = {
        "mode": "file",
        "file_path": str(csv_path),
        "allowed_skew_seconds": 600,
        "interpolation": "nearest",
    }
    targets = [
        "2024-01-01T00:04:00Z",
        "2024-01-01T00:16:00Z",
    ]

    result = load_external_rn_series(cfg, targets)

    assert [round(val, 3) for _, val in result] == [60.0, 120.0]


def test_file_mode_ffill(tmp_path):
    csv_path = tmp_path / "external_ffill.csv"
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:10:00",
            ],
            "rn_bq_per_m3": [50.0, 75.0],
        }
    )
    df.to_csv(csv_path, index=False)

    cfg = {
        "mode": "file",
        "file_path": str(csv_path),
        "timezone": "America/Denver",
        "allowed_skew_seconds": 900,
        "interpolation": "ffill",
    }
    targets = [
        "2024-01-01T07:05:00Z",
        "2024-01-01T07:15:00Z",
    ]

    result = load_external_rn_series(cfg, targets)

    assert [round(val, 3) for _, val in result] == [50.0, 75.0]


def test_file_mode_missing_values_fallback_to_constant(tmp_path):
    csv_path = tmp_path / "external_sparse.csv"
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:30:00Z",
            ],
            "rn_bq_per_m3": [55.0, 65.0],
        }
    )
    df.to_csv(csv_path, index=False)

    cfg = {
        "mode": "file",
        "file_path": str(csv_path),
        "allowed_skew_seconds": 60,  # 1 minute tolerance forces fallback
        "constant_bq_per_m3": 88.8,
    }
    targets = [
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:15:00Z",
        "2024-01-01T00:30:00Z",
    ]

    result = load_external_rn_series(cfg, targets)

    assert [round(val, 3) for _, val in result] == [55.0, 88.8, 65.0]


def test_file_mode_missing_file_without_constant_raises(tmp_path):
    cfg = {
        "mode": "file",
        "file_path": str(tmp_path / "missing.csv"),
    }

    with pytest.raises(FileNotFoundError) as excinfo:
        load_external_rn_series(cfg, ["2024-01-01T00:00:00Z"])

    assert "external radon file not found" in str(excinfo.value)


def test_file_mode_missing_file_with_constant_falls_back(tmp_path):
    cfg = {
        "mode": "file",
        "file_path": str(tmp_path / "missing.csv"),
        "constant_bq_per_m3": 101.0,
    }

    result = load_external_rn_series(cfg, ["2024-01-01T00:00:00Z"])

    assert [val for _, val in result] == [101.0]
