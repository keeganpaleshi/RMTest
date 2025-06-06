import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import cps_to_cpd, cps_to_bq


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
