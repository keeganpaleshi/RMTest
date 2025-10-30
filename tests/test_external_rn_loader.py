import pandas as pd
import pytest

from io_utils import load_external_rn_series


def test_load_external_rn_series_constant():
    timestamps = ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"]
    cfg = {"mode": "constant", "constant_bq_per_m3": 123.4}

    result = load_external_rn_series(cfg, timestamps)

    assert [float(value) for _, value in result] == [123.4, 123.4]


def test_load_external_rn_series_file_nearest(tmp_path):
    timestamps = [
        "2024-01-01T00:01:00Z",
        "2024-01-01T00:06:00Z",
        "2024-01-01T00:11:00Z",
    ]

    data = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:05:00Z",
                "2024-01-01T00:10:00Z",
            ],
            "rn_bq_per_m3": [80, 100, 120],
        }
    )
    csv_path = tmp_path / "external.csv"
    data.to_csv(csv_path, index=False)

    cfg = {
        "mode": "file",
        "file_path": str(csv_path),
        "interpolation": "nearest",
        "allowed_skew_seconds": 300,
    }

    result = load_external_rn_series(cfg, timestamps)

    assert [value for _, value in result] == [80.0, 100.0, 120.0]


def test_load_external_rn_series_file_constant_fallback(tmp_path):
    timestamps = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]

    data = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "rn_bq_per_m3": [50],
        }
    )
    csv_path = tmp_path / "external.csv"
    data.to_csv(csv_path, index=False)

    cfg = {
        "mode": "file",
        "file_path": str(csv_path),
        "interpolation": "ffill",
        "allowed_skew_seconds": 60,
        "constant_bq_per_m3": 75,
    }

    result = load_external_rn_series(cfg, timestamps)

    assert [value for _, value in result] == [50.0, 75.0]


def test_load_external_rn_series_missing_file_without_constant(tmp_path):
    cfg = {"mode": "file", "file_path": str(tmp_path / "missing.csv"), "constant_bq_per_m3": None}

    with pytest.raises(FileNotFoundError) as excinfo:
        load_external_rn_series(cfg, ["2024-01-01T00:00:00Z"])

    assert "external radon file not found" in str(excinfo.value)
