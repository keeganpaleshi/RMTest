import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rmtest.radon_inference import run_radon_inference


def _basic_config(**overrides):
    cfg = {
        "radon_inference": {
            "enabled": True,
            "source_isotopes": ["Po214"],
            "detection_efficiency": {"Po214": 0.12},
            "transport_efficiency": 1.0,
            "retention_efficiency": 1.0,
            "external_rn": {"mode": "constant", "constant_bq_per_m3": 80.0},
        }
    }
    cfg["radon_inference"].update(overrides)
    return cfg


def test_constant_external_radon_computes_expected_activity():
    isotope_series = {"Po214": [{"t": 0.0, "counts": 120.0, "dt": 60.0}]}
    cfg = _basic_config()

    result = run_radon_inference(isotope_series, cfg)

    assert result is not None
    rn_points = result["rn_inferred"]
    assert pytest.approx(rn_points[0]["rn_bq"], rel=1e-6) == 120.0 / (0.12 * 60.0)

    volume = result["volume_equiv"][0]
    assert pytest.approx(volume["v_m3"], rel=1e-6) == (120.0 / (0.12 * 60.0)) * 60.0 / 80.0
    assert pytest.approx(volume["v_lpm"], rel=1e-6) == volume["v_m3"] * 1000.0


def test_missing_isotope_reweights_available_sources():
    isotope_series = {"Po214": [{"t": 0.0, "counts": 50.0, "dt": 30.0}]}
    cfg = _basic_config(
        source_isotopes=["Po214", "Po218"],
        source_weights={"Po214": 0.7, "Po218": 0.3},
        detection_efficiency={"Po214": 0.2, "Po218": 0.15},
    )

    result = run_radon_inference(isotope_series, cfg)

    assert result is not None
    weights = result["meta"]["source_weights"]
    assert list(weights.keys()) == ["Po214"]
    assert pytest.approx(weights["Po214"], rel=1e-6) == 1.0


def test_external_file_with_missing_values_falls_back_to_constant():
    fixture = Path(__file__).resolve().parent / "data" / "mine_rn_timeseries.csv"
    cfg = {
        "radon_inference": {
            "enabled": True,
            "source_isotopes": ["Po214"],
            "detection_efficiency": {"Po214": 0.25},
            "transport_efficiency": 1.0,
            "retention_efficiency": 1.0,
            "external_rn": {
                "mode": "file",
                "file_path": str(fixture),
                "time_column": "timestamp",
                "value_column": "rn_bq_per_m3",
                "interpolation": "nearest",
                "allowed_skew_seconds": 10,
                "constant_bq_per_m3": 70.0,
            },
        }
    }

    timestamps = [0.0, 60.0, 120.0, 180.0, 240.0]
    isotope_series = {
        "Po214": [{"t": ts, "counts": 25.0, "dt": 60.0} for ts in timestamps]
    }

    result = run_radon_inference(isotope_series, cfg)

    assert result is not None
    ambient = result["ambient_rn"]
    assert len(ambient) == len(timestamps)
    assert all(entry["rn_bq_per_m3"] is not None for entry in ambient)

    vols = result["volume_equiv"]
    assert len(vols) == len(timestamps)
    cumulative = result["volume_cumulative"]
    assert cumulative
    assert math.isclose(cumulative[-1]["v_m3_cum"], sum(v["v_m3"] for v in vols))
