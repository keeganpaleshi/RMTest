import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import cps_to_cpd, cps_to_bq, find_adc_bin_peaks


def test_cps_to_cpd():
    assert cps_to_cpd(1.0) == pytest.approx(86400.0)


def test_cps_to_bq_volume():
    # 2 cps in 10 L -> 2/(0.01 m^3) = 200 Bq/m^3
    assert cps_to_bq(2.0, volume_liters=10.0) == pytest.approx(200.0)


def test_cps_to_bq_simple():
    assert cps_to_bq(3.5) == pytest.approx(3.5)


def test_cps_to_bq_zero_volume():
    with pytest.raises(ValueError):
        cps_to_bq(1.0, volume_liters=0)


def test_cps_to_bq_negative_volume():
    with pytest.raises(ValueError):
        cps_to_bq(1.0, volume_liters=-1.0)


def test_find_adc_bin_peaks_simple():
    adc = [5, 10, 10, 10, 15, 20, 20, 20, 25]
    expected = {"peak1": 10, "peak2": 20}
    res = find_adc_bin_peaks(adc, expected, window=3)
    assert set(res) == {"peak1", "peak2"}
    assert pytest.approx(res["peak1"], rel=1e-3) == 10.5
    assert pytest.approx(res["peak2"], rel=1e-3) == 20.5
