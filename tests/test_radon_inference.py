from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

from radon.external_rn_loader import load_external_rn_series
from radon.radon_inference import run_radon_inference


def _config(**overrides):
    cfg = {
        "radon_inference": {
            "enabled": True,
            "source_isotopes": ["Po214"],
            "source_weights": {"Po214": 1.0},
            "detection_efficiency": {"Po214": 0.12},
            "transport_efficiency": 1.0,
            "retention_efficiency": 1.0,
            "chain_correction": "none",
        }
    }
    cfg["radon_inference"].update(overrides)
    return cfg


def test_radon_inference_constant_external_rn():
    iso_series = {"Po214": [{"t": 0.0, "counts": 120.0, "dt": 60.0}]}
    external = [(datetime(2024, 1, 1, tzinfo=timezone.utc), 80.0)]

    result = run_radon_inference(iso_series, _config(), external_rn_series=external)

    rn_series = result.get("rn_inferred")
    assert rn_series and math.isclose(rn_series[0]["rn_bq"], 120.0 / (0.12 * 60.0), rel_tol=1e-9)

    volumes = result.get("volume_equiv")
    assert volumes and math.isclose(volumes[0]["v_m3"], (120.0 / (0.12 * 60.0)) * 60.0 / 80.0, rel_tol=1e-9)
    assert math.isclose(volumes[0]["v_lpm"], 12500.0, rel_tol=1e-6)


def test_radon_inference_missing_isotope_weight_normalisation():
    iso_series = {"Po214": [{"t": 0.0, "counts": 10.0, "dt": 60.0}]}
    cfg = _config(
        source_isotopes=["Po214", "Po218"],
        source_weights={"Po214": 0.4, "Po218": 0.6},
        detection_efficiency={"Po214": 0.12, "Po218": 0.11},
    )

    result = run_radon_inference(iso_series, cfg)

    meta = result.get("meta")
    assert meta is not None
    assert meta["source_isotopes"] == ["Po214"]
    assert math.isclose(meta["source_weights"]["Po214"], 1.0)


def test_external_rn_file_reindex(tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=i) for i in range(5)]
    series = [
        {"t": ts.timestamp(), "counts": 60.0, "dt": 60.0}
        for ts in timestamps
    ]

    cfg_external = {
        "mode": "file",
        "file_path": tmp_path / "mine_rn_timeseries.csv",
        "time_column": "timestamp",
        "value_column": "rn_bq_per_m3",
        "timezone": "UTC",
        "interpolation": "nearest",
        "allowed_skew_seconds": 600,
        "default_bq_per_m3": 80.0,
    }

    fixture = Path(__file__).with_name("data") / "mine_rn_timeseries.csv"
    cfg_external["file_path"].write_text(fixture.read_text())

    ambient = load_external_rn_series(cfg_external, [ts.timestamp() for ts in timestamps])
    assert len(ambient) == len(timestamps)

    iso_series = {"Po214": series}
    cfg = _config()

    result = run_radon_inference(iso_series, cfg, external_rn_series=ambient)

    rn_series = result.get("rn_inferred")
    assert rn_series and len(rn_series) == len(timestamps)
    ambient_series = result.get("ambient_rn")
    assert ambient_series and len(ambient_series) == len(timestamps)
    volumes = result.get("volume_equiv")
    assert volumes and len(volumes) == len(timestamps)
