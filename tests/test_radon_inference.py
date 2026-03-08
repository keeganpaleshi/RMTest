import math
from pathlib import Path

import pytest

from constants import RN222
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


def _rn222_lambda() -> float:
    return math.log(2.0) / RN222.half_life_s


def test_constant_external_rn_volume_math():
    config = _base_config(
        external_rn={"mode": "constant", "constant_bq_per_m3": 80.0}
    )
    series = {"Po214": [{"t": 30.0, "counts": 120.0, "dt": 60.0}]}
    external = [{"t": 30.0, "rn_bq_per_m3": 80.0}]

    result = run_radon_inference(series, config, external)
    assert result is not None

    expected_activity = 120.0 / (0.12 * 60.0)
    expected_rate = _rn222_lambda() * expected_activity / 80.0
    expected_volume = expected_rate * 60.0

    rn_entry = result["rn_inferred"][0]
    assert rn_entry["rn_bq"] == pytest.approx(expected_activity)

    vol_entry = result["volume_equiv"][0]
    assert vol_entry["q_m3_s"] == pytest.approx(expected_rate)
    assert vol_entry["v_m3"] == pytest.approx(expected_volume)
    assert vol_entry["v_lpm"] == pytest.approx(expected_rate * 60000.0)
    assert result["volume_cumulative"][0]["v_m3_cum"] == pytest.approx(expected_volume)


def test_leak_volume_is_stable_under_rebinning_for_steady_state():
    config = _base_config(
        external_rn={"mode": "constant", "constant_bq_per_m3": 80.0}
    )
    leak_rate = 1.25e-6
    ambient = 80.0
    dt_split = 60.0
    dt_merged = 120.0
    efficiency = 0.12
    activity = leak_rate * ambient / _rn222_lambda()

    split_series = {
        "Po214": [
            {"t": 30.0, "counts": activity * efficiency * dt_split, "dt": dt_split},
            {"t": 90.0, "counts": activity * efficiency * dt_split, "dt": dt_split},
        ]
    }
    merged_series = {
        "Po214": [{"t": 60.0, "counts": activity * efficiency * dt_merged, "dt": dt_merged}]
    }

    split_result = run_radon_inference(
        split_series,
        config,
        [
            {"t": 30.0, "rn_bq_per_m3": ambient},
            {"t": 90.0, "rn_bq_per_m3": ambient},
        ],
    )
    merged_result = run_radon_inference(
        merged_series,
        config,
        [{"t": 60.0, "rn_bq_per_m3": ambient}],
    )

    assert split_result is not None
    assert merged_result is not None

    split_volumes = [entry["v_m3"] for entry in split_result["volume_equiv"]]
    assert split_volumes == pytest.approx([leak_rate * dt_split, leak_rate * dt_split])
    assert split_result["volume_equiv"][0]["q_m3_s"] == pytest.approx(leak_rate)
    assert split_result["volume_equiv"][1]["q_m3_s"] == pytest.approx(leak_rate)
    assert merged_result["volume_equiv"][-1]["v_m3"] == pytest.approx(
        leak_rate * dt_merged
    )
    assert merged_result["volume_equiv"][-1]["q_m3_s"] == pytest.approx(leak_rate)
    assert merged_result["volume_cumulative"][-1]["v_m3_cum"] == pytest.approx(
        split_result["volume_cumulative"][-1]["v_m3_cum"]
    )


def test_leak_volume_recovers_constant_rate_with_changing_ambient():
    config = _base_config(
        external_rn={"mode": "constant", "constant_bq_per_m3": 80.0}
    )
    leak_rate = 1.0e-6
    dt = 60.0
    efficiency = 0.12
    ambient0 = 80.0
    ambient1 = 160.0
    lambda_rn = _rn222_lambda()
    activity0 = leak_rate * ambient0 / lambda_rn
    survival = math.exp(-lambda_rn * dt)
    response = (1.0 - survival) / lambda_rn
    mean_weight = response / dt
    leak_weight = 1.0 - mean_weight
    activity1 = activity0 * mean_weight + leak_rate * ambient1 * leak_weight / lambda_rn

    series = {
        "Po214": [
            {"t": 30.0, "counts": activity0 * efficiency * dt, "dt": dt},
            {"t": 90.0, "counts": activity1 * efficiency * dt, "dt": dt},
        ]
    }
    external = [
        {"t": 30.0, "rn_bq_per_m3": ambient0},
        {"t": 90.0, "rn_bq_per_m3": ambient1},
    ]

    result = run_radon_inference(series, config, external)
    assert result is not None

    volumes = [entry["v_m3"] for entry in result["volume_equiv"]]
    rates = [entry["q_m3_s"] for entry in result["volume_equiv"]]
    cumulative = [entry["v_m3_cum"] for entry in result["volume_cumulative"]]
    assert volumes == pytest.approx([leak_rate * dt, leak_rate * dt])
    assert rates == pytest.approx([leak_rate, leak_rate])
    assert cumulative == pytest.approx([leak_rate * dt, 2.0 * leak_rate * dt])


def test_negative_leak_rates_are_clipped_to_zero():
    config = _base_config(
        external_rn={"mode": "constant", "constant_bq_per_m3": 80.0}
    )
    dt = 60.0
    efficiency = 0.12
    activity0 = 10.0
    activity1 = 4.0
    series = {
        "Po214": [
            {"t": 30.0, "counts": activity0 * efficiency * dt, "dt": dt},
            {"t": 90.0, "counts": activity1 * efficiency * dt, "dt": dt},
        ]
    }
    external = [
        {"t": 30.0, "rn_bq_per_m3": 80.0},
        {"t": 90.0, "rn_bq_per_m3": 80.0},
    ]

    result = run_radon_inference(series, config, external)
    assert result is not None

    first, second = result["volume_equiv"]
    assert first["v_m3"] > 0.0
    assert second["q_m3_s"] == pytest.approx(0.0)
    assert second["v_m3"] == pytest.approx(0.0)
    assert second["meta"]["clipped_to_zero"] is True
    assert result["volume_cumulative"][-1]["v_m3_cum"] == pytest.approx(
        result["volume_cumulative"][0]["v_m3_cum"]
    )


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
