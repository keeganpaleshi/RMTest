from pathlib import Path

import math

import pytest

from radon_inference import run_radon_inference
from radon.external_rn_loader import load_external_rn_series


def _base_config(**overrides):
    cfg = {
        "radon_inference": {
            "enabled": True,
            "source_isotopes": ["Po214", "Po218"],
            "source_weights": {"Po214": 0.7, "Po218": 0.3},
            "detection_efficiency": {"Po214": 0.12, "Po218": 0.1},
            "transport_efficiency": 1.0,
            "retention_efficiency": 1.0,
            "chain_correction": "none",
        }
    }
    cfg["radon_inference"].update(overrides)
    return cfg


def test_constant_external_rn_volume_math():
    cfg = _base_config()
    isotope_series = {
        "Po214": [
            {
                "t": "2024-01-01T00:00:00Z",
                "counts": 120.0,
                "dt": 60.0,
            }
        ]
    }
    external = [("2024-01-01T00:00:00Z", 80.0)]

    result = run_radon_inference(isotope_series, cfg, external)
    assert result is not None

    expected_activity = 120.0 / (0.12 * 60.0)
    assert result["rn_inferred"][0]["rn_bq"] == pytest.approx(expected_activity)

    volume_entry = 12.5  # m^3
    volume = result["volume_equiv"][0]
    assert volume["v_m3"] == pytest.approx(volume_entry)
    assert volume["v_lpm"] == pytest.approx(12500.0)


def test_missing_isotope_renormalizes_weights():
    cfg = _base_config()
    cfg["radon_inference"]["source_isotopes"] = ["Po214", "Po218"]
    isotope_series = {
        "Po214": [
            {"t": "2024-01-01T00:00:00Z", "counts": 50.0, "dt": 60.0},
        ]
    }

    result = run_radon_inference(isotope_series, cfg, external_rn_series=None)
    assert result is not None

    weights = result["rn_inferred"][0]["meta"]["weights"]
    assert set(weights.keys()) == {"Po214"}
    assert math.isclose(weights["Po214"], 1.0, rel_tol=1e-12)


def test_external_file_missing_values_falls_back():
    fixture = Path(__file__).resolve().parent / "data" / "mine_rn_timeseries.csv"
    external_cfg = {
        "mode": "file",
        "file_path": str(fixture),
        "interpolation": "nearest",
        "allowed_skew_seconds": 600,
        "default_bq_per_m3": 85.0,
    }

    cfg = _base_config(external_rn=external_cfg)

    timestamps = [
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:05:00Z",
        "2024-01-01T00:10:00Z",
        "2024-01-01T00:15:00Z",
        "2024-01-01T00:20:00Z",
    ]
    isotope_series = {
        "Po214": [
            {"t": ts, "counts": 40.0 + idx, "dt": 60.0}
            for idx, ts in enumerate(timestamps)
        ]
    }

    external_series = load_external_rn_series(external_cfg, timestamps)
    result = run_radon_inference(isotope_series, cfg, external_series)
    assert result is not None

    assert len(result["volume_equiv"]) == len(timestamps)
    assert all(entry["v_m3_cum"] >= 0 for entry in result["volume_cumulative"])

