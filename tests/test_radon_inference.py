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


def test_dedupe_identical_entries():
    """Test 1: Dedupe works for identical Po214 entries at same timestamp."""
    config = _base_config(detection_efficiency={"Po214": 0.12})
    # Two identical entries at the same timestamp
    series = {
        "Po214": [
            {"t": 100.0, "counts": 50.0, "dt": 60.0},
            {"t": 100.0, "counts": 50.0, "dt": 60.0},
        ]
    }

    result = run_radon_inference(series, config)
    assert result is not None

    rn_inferred = result["rn_inferred"]
    # Should have only 1 entry after deduplication
    assert len(rn_inferred) == 1
    assert rn_inferred[0]["t"] == pytest.approx(100.0)
    # Verify the calculation: A_iso = counts / (eff * dt) = 50.0 / (0.12 * 60.0)
    expected_rn_bq = 50.0 / (0.12 * 60.0)
    assert rn_inferred[0]["rn_bq"] == pytest.approx(expected_rn_bq)


def test_overlay_dedupe():
    """Test 2: Dedupe works with overlay - same isotope from multiple sources."""
    config = _base_config(detection_efficiency={"Po214": 0.12})
    # Simulate Po214 from source A and source B at same timestamp
    # In practice, these would come from different plot sources but end up
    # in the same isotope_series_data["Po214"] list
    series = {
        "Po214": [
            {"t": 200.0, "counts": 75.0, "dt": 60.0},  # source A
            {"t": 200.0, "counts": 75.0, "dt": 60.0},  # source B (identical)
        ]
    }

    result = run_radon_inference(series, config)
    assert result is not None

    rn_inferred = result["rn_inferred"]
    # Should have only 1 Po214 entry after deduplication
    assert len(rn_inferred) == 1
    assert rn_inferred[0]["t"] == pytest.approx(200.0)
    # Verify it's not double-counted
    expected_rn_bq = 75.0 / (0.12 * 60.0)
    assert rn_inferred[0]["rn_bq"] == pytest.approx(expected_rn_bq)


def test_external_rn_ffill():
    """Test 3: External radon loader forward-fills sparse measurements."""
    # Target timestamps every 60s
    target_timestamps = [0.0, 60.0, 120.0, 180.0, 240.0]

    # External CSV has entries every 300s (5 minutes)
    # We'll simulate this with a config that provides sparse data
    import tempfile
    import pandas as pd

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Write CSV with entries at 0, 300, 600
        f.write("timestamp,rn_bq_per_m3\n")
        f.write("1970-01-01T00:00:00Z,50.0\n")
        f.write("1970-01-01T00:05:00Z,60.0\n")
        f.write("1970-01-01T00:10:00Z,70.0\n")
        csv_path = f.name

    try:
        cfg_external = {
            "mode": "file",
            "file_path": csv_path,
            "interpolation": "ffill",
            "allowed_skew_seconds": 300,
            "constant_bq_per_m3": 100.0,
        }

        result = load_external_rn_series(cfg_external, target_timestamps)
        assert len(result) == 5

        # Check forward-fill behavior:
        # t=0s should get 50.0
        # t=60s, 120s, 180s, 240s should all forward-fill from 50.0
        times, values = zip(*result)
        assert values[0] == pytest.approx(50.0)  # t=0
        assert values[1] == pytest.approx(50.0)  # t=60 (ffill from 0)
        assert values[2] == pytest.approx(50.0)  # t=120 (ffill from 0)
        assert values[3] == pytest.approx(50.0)  # t=180 (ffill from 0)
        assert values[4] == pytest.approx(50.0)  # t=240 (ffill from 0)

    finally:
        Path(csv_path).unlink()


def test_external_rn_fallback():
    """Test 4: External radon loader uses fallback for out-of-range timestamps."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # CSV has data from t=100 to t=200
        f.write("timestamp,rn_bq_per_m3\n")
        f.write("1970-01-01T00:01:40Z,80.0\n")  # t=100s
        f.write("1970-01-01T00:03:20Z,90.0\n")  # t=200s
        csv_path = f.name

    try:
        cfg_external = {
            "mode": "file",
            "file_path": csv_path,
            "interpolation": "nearest",
            "allowed_skew_seconds": 60,
            "constant_bq_per_m3": 100.0,
        }

        # Target timestamps outside CSV range
        target_timestamps = [0.0, 50.0, 150.0, 250.0, 300.0]
        result = load_external_rn_series(cfg_external, target_timestamps)
        assert len(result) == 5

        times, values = zip(*result)
        # t=0, t=300 should use fallback (100.0) because they're > 60s away from any CSV entry
        # t=50 should get 80.0 from t=100 (50s away, within tolerance)
        # t=150 should get either 80.0 or 90.0 (both 50s away, within tolerance)
        # t=250 should get 90.0 from t=200 (50s away, within tolerance)
        assert values[0] == pytest.approx(100.0)  # t=0 (100s from nearest, use fallback)
        assert values[1] == pytest.approx(80.0)   # t=50 (50s from t=100, within tolerance)
        assert values[2] == pytest.approx(80.0) or values[2] == pytest.approx(90.0)  # t=150 (50s from both)
        assert values[3] == pytest.approx(90.0)   # t=250 (50s from t=200, within tolerance)
        assert values[4] == pytest.approx(100.0)  # t=300 (100s from nearest, use fallback)

    finally:
        Path(csv_path).unlink()
