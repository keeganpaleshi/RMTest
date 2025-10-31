import math
from pathlib import Path

import pytest

from radon.external_rn_loader import load_external_rn_series
from radon.radon_inference import run_radon_inference


def _base_config(**overrides):
    cfg = {
        "radon_inference": {
            "enabled": True,
            "source_isotopes": ["Po214"],
            "detection_efficiency": {"Po214": 0.12},
            "transport_efficiency": 1.0,
            "retention_efficiency": 1.0,
            "chain_correction": "none",
        }
    }
    cfg["radon_inference"].update(overrides)
    return cfg


def test_constant_external_rn_volume_math():
    config = _base_config(
        external_rn={"mode": "constant", "constant_bq_per_m3": 80.0}
    )
    series = {"Po214": [{"t": 0.0, "counts": 120.0, "dt": 60.0}]}
    external = [{"t": 0.0, "rn_bq_per_m3": 80.0}]

    result = run_radon_inference(series, config, external)
    assert result is not None

    rn_entry = result["rn_inferred"][0]
    assert rn_entry["rn_bq"] == pytest.approx(120.0 / (0.12 * 60.0))

    vol_entry = result["volume_equiv"][0]
    assert vol_entry["v_m3"] == pytest.approx(12.5)
    assert vol_entry["v_lpm"] == pytest.approx(12500.0)


def test_missing_isotope_reweights_po214_only():
    config = _base_config(
        source_isotopes=["Po214", "Po218"],
        source_weights={"Po214": 0.7, "Po218": 0.3},
        detection_efficiency={"Po214": 0.12, "Po218": 0.1},
    )
    series = {"Po214": [{"t": 0.0, "counts": 50.0, "dt": 60.0}]}

    result = run_radon_inference(series, config)
    assert result is not None

    meta = result["meta"]
    assert meta["source_isotopes"] == ["Po214"]
    assert meta["source_weights"]["Po214"] == pytest.approx(1.0)


def test_external_file_with_sparse_samples():
    data_file = Path(__file__).parent / "data" / "mine_rn_timeseries.csv"
    config = _base_config(
        detection_efficiency={"Po214": 0.5},
        external_rn={
            "mode": "file",
            "file_path": str(data_file),
            "interpolation": "nearest",
            "allowed_skew_seconds": 10,
            "default_bq_per_m3": 70.0,
            "constant_bq_per_m3": 70.0,
        },
    )

    times = [0.0, 60.0, 120.0, 180.0, 240.0]
    series = {
        "Po214": [
            {"t": t, "counts": 30.0 + idx, "dt": 60.0}
            for idx, t in enumerate(times)
        ]
    }

    external = load_external_rn_series(config["radon_inference"]["external_rn"], times)
    result = run_radon_inference(series, config, external)
    assert result is not None

    ambient = result["ambient_rn"]
    assert len(ambient) == len(times)
    ambient_map = {round(entry["t"], 6): entry["rn_bq_per_m3"] for entry in ambient}
    # Values at 60s and 180s fall back to the default because of the tight tolerance
    assert ambient_map[60.0] == pytest.approx(70.0)
    assert ambient_map[180.0] == pytest.approx(70.0)

    volumes = result["volume_equiv"]
    assert volumes  # ensures equivalent volume still produced
    assert all(math.isfinite(entry["v_m3"]) for entry in volumes)
