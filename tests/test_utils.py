import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import cps_to_cpd, cps_to_bq, find_adc_bin_peaks, parse_time, LITERS_PER_M3


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


def test_parse_time_int():
    assert parse_time(42) == pytest.approx(42.0)


def test_parse_time_float():
    assert parse_time(42.5) == pytest.approx(42.5)


def test_parse_time_numeric_str():
    assert parse_time("42") == pytest.approx(42.0)


def test_parse_time_numeric_str_float():
    assert parse_time("42.5") == pytest.approx(42.5)


def test_parse_time_iso_no_fraction():
    assert parse_time("1970-01-01T00:00:00Z") == pytest.approx(0.0)


def test_parse_time_iso_fraction():
    assert parse_time("1970-01-01T00:00:00.5Z") == pytest.approx(0.5)
