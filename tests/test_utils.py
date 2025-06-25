import sys
from pathlib import Path
from datetime import datetime, timezone
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (
    cps_to_cpd,
    cps_to_bq,
    find_adc_bin_peaks,
    parse_time_arg,
    to_seconds,
    LITERS_PER_M3,
)
from time_utils import parse_timestamp, to_epoch_seconds


def test_cps_to_cpd():
    assert cps_to_cpd(1.0) == pytest.approx(86400.0)


def test_cps_to_bq_volume():
    # 2 cps in 10 L -> 2/(10/LITERS_PER_M3) Bq/m^3
    expected = 2.0 / (10.0 / LITERS_PER_M3)
    assert cps_to_bq(2.0, volume_liters=10.0) == pytest.approx(expected)


def test_cps_to_bq_simple():
    assert cps_to_bq(3.5) == pytest.approx(3.5)


def test_cps_to_bq_zero_volume():
    with pytest.raises(ValueError):
        cps_to_bq(1.0, volume_liters=0)


def test_cps_to_bq_negative_volume():
    with pytest.raises(ValueError):
        cps_to_bq(1.0, volume_liters=-1.0)


def test_find_adc_bin_peaks_basic():
    adc = [10, 10, 20, 20, 20, 30]
    expected = {"p1": 10, "p2": 20}
    result = find_adc_bin_peaks(adc, expected, window=2)
    assert result["p1"] == pytest.approx(10.5)
    assert result["p2"] == pytest.approx(20.5)


def test_parse_timestamp_int():
    assert parse_timestamp(42) == pytest.approx(42.0)


def test_parse_timestamp_float():
    assert parse_timestamp(42.5) == pytest.approx(42.5)


def test_parse_timestamp_numeric_str():
    assert parse_timestamp("42") == pytest.approx(42.0)


def test_parse_timestamp_numeric_str_float():
    assert parse_timestamp("42.5") == pytest.approx(42.5)


def test_parse_timestamp_iso_no_fraction():
    assert parse_timestamp("1970-01-01T00:00:00Z") == pytest.approx(0.0)


def test_parse_timestamp_iso_fraction():
    assert parse_timestamp("1970-01-01T00:00:00.5Z") == pytest.approx(0.5)


def test_parse_time_naive_timezone():
    assert parse_time_arg("1970-01-01T01:00:00", tz="Europe/Berlin") == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_parse_timestamp_numeric():
    assert parse_timestamp(42) == pytest.approx(42.0)


def test_parse_timestamp_iso():
    assert parse_timestamp("1970-01-01T00:00:00Z") == pytest.approx(0.0)


def test_parse_timestamp_datetime_naive():
    assert parse_timestamp(datetime(1970, 1, 1)) == pytest.approx(0.0)


def test_to_seconds_datetime_series():
    ser = pd.Series(pd.to_datetime([0, 1, 2], unit="s", utc=True))
    out = to_seconds(ser)
    assert np.allclose(out, [0.0, 1.0, 2.0])


def test_to_seconds_numeric_series():
    ser = pd.Series([0.0, 1.5, 2.2])
    out = to_seconds(ser)
    assert np.allclose(out, [0.0, 1.5, 2.2])


def test_to_epoch_seconds_scalar():
    out = to_epoch_seconds("1970-01-01T00:00:01Z")
    assert np.allclose(out, [1.0])
