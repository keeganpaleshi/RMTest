from datetime import datetime, timezone

import pytest

from io_utils import DEFAULT_EXTERNAL_RN_BQ_PER_M3, load_external_rn_series


def _timestamps(*parts):
    return [datetime(*p, tzinfo=timezone.utc) for p in parts]


def test_load_external_rn_series_constant_default():
    target = _timestamps((2023, 1, 1, 0, 0, 0), (2023, 1, 1, 1, 0, 0))
    cfg = {"mode": "constant"}

    aligned = load_external_rn_series(cfg, target)

    assert [round(v, 6) for _, v in aligned] == [DEFAULT_EXTERNAL_RN_BQ_PER_M3] * len(target)
    assert [ts for ts, _ in aligned] == target


def test_load_external_rn_series_file_nearest_timezone(tmp_path):
    csv_path = tmp_path / "external_rn.csv"
    csv_path.write_text(
        "time,value\n"
        "2023-01-01 00:00,70\n"
        "2023-01-01 01:00,90\n"
        "2023-01-01 02:00,110\n"
    )

    cfg = {
        "mode": "file",
        "file_path": csv_path,
        "time_column": "time",
        "value_column": "value",
        "timezone": "US/Eastern",
        "interpolation": "nearest",
        "allowed_skew_seconds": 600,
    }

    target = _timestamps(
        (2023, 1, 1, 5, 0, 0),
        (2023, 1, 1, 6, 0, 0),
        (2023, 1, 1, 7, 0, 0),
    )

    aligned = load_external_rn_series(cfg, target)

    assert [round(v, 6) for _, v in aligned] == [70.0, 90.0, 110.0]


def test_load_external_rn_series_file_ffill_with_constant_fallback(tmp_path):
    csv_path = tmp_path / "external_ffill.csv"
    csv_path.write_text(
        "time,value\n"
        "2023-01-01T05:00:00Z,60\n"
        "2023-01-01T06:00:00Z,80\n"
    )

    cfg = {
        "mode": "file",
        "file_path": csv_path,
        "time_column": "time",
        "value_column": "value",
        "interpolation": "ffill",
        "allowed_skew_seconds": 5400,
        "constant_bq_per_m3": 200.0,
    }

    target = _timestamps(
        (2023, 1, 1, 5, 0, 0),
        (2023, 1, 1, 6, 30, 0),
        (2023, 1, 1, 8, 0, 0),
    )

    aligned = load_external_rn_series(cfg, target)

    assert [round(v, 6) for _, v in aligned] == [60.0, 80.0, 200.0]


def test_load_external_rn_series_missing_file_without_constant(tmp_path):
    cfg = {
        "mode": "file",
        "file_path": tmp_path / "missing.csv",
        "interpolation": "nearest",
        "allowed_skew_seconds": 60,
        "default_bq_per_m3": None,
        "constant_bq_per_m3": None,
    }

    with pytest.raises(FileNotFoundError) as exc:
        load_external_rn_series(cfg, _timestamps((2023, 1, 1, 0, 0, 0)))

    assert "external radon file not found" in str(exc.value)


def test_load_external_rn_series_missing_file_with_constant(tmp_path):
    cfg = {
        "mode": "file",
        "file_path": tmp_path / "missing.csv",
        "constant_bq_per_m3": 123.0,
    }

    aligned = load_external_rn_series(cfg, _timestamps((2023, 1, 1, 0, 0, 0)))

    assert [round(v, 6) for _, v in aligned] == [123.0]
