import pytest

from baseline_handling import (
    BaselineRecord,
    ingest_baseline_record,
    get_fixed_background_for_time_fit,
    annotate_time_fit_with_baseline,
    evaluate_baseline_drift,
    apply_baseline_drift_warning,
)


@pytest.fixture
def baseline_dict():
    return {
        "start": "2024-09-28T00:00:00Z",
        "end": "2024-09-29T00:00:00Z",
        "live_time": 86400.0,
        "rate_Bq": {"Po214": 0.12},
        "rate_unc_Bq": {"Po214": 0.01},
        "dilution_factor": 1.0,
        "calibration_snapshot": {
            "po214_centroid_mev": 7.69,
            "sigma_E": 0.12,
        },
    }


def test_ingest_baseline_record_parses_fields(baseline_dict):
    record = ingest_baseline_record(baseline_dict)
    assert isinstance(record, BaselineRecord)
    assert record.live_time_s == pytest.approx(86400.0)
    assert record.get_rate("Po214") == pytest.approx(0.12)
    assert record.get_uncertainty("Po214") == pytest.approx(0.01)
    assert record.time_range == (
        "2024-09-28T00:00:00Z",
        "2024-09-29T00:00:00Z",
    )


def test_get_fixed_background_scales_rate(baseline_dict):
    record = ingest_baseline_record(baseline_dict)
    cfg = {"baseline": {"monitor_volume_l": 605.0, "sample_volume_l": 605.0}}
    result = get_fixed_background_for_time_fit(record, "Po214", cfg)
    assert result is not None
    assert result["background_rate_Bq"] == pytest.approx(0.12 * 0.5)
    assert result["background_unc_Bq"] == pytest.approx(0.01 * 0.5)
    assert result["mode"] == "baseline_fixed"


def test_annotate_time_fit_with_baseline_adds_provenance(baseline_dict):
    record = ingest_baseline_record(baseline_dict)
    summary = {"time_fit": {"Po214": {}}, "baseline": {}}
    provenance = {"Po214": {"background_rate_Bq": 0.05, "mode": "baseline_fixed"}}
    annotate_time_fit_with_baseline(summary, record, provenance)
    entry = summary["time_fit"]["Po214"]
    assert entry["background_source"] == "baseline_fixed"
    assert entry["baseline_source_range"] == list(record.time_range)
    assert entry["baseline_activity_Bq"] == pytest.approx(0.05)
    assert "record" in summary["baseline"]


def test_evaluate_baseline_drift_flags_warning(baseline_dict):
    record = ingest_baseline_record(baseline_dict)
    calibration = {
        "peaks": {"Po214": {"centroid_mev": 7.80}},
        "sigma_E": 0.15,
    }
    warning, message = evaluate_baseline_drift(record, calibration)
    assert warning is True
    assert "drifted" in message


def test_apply_baseline_drift_warning_updates_summary():
    summary = {"diagnostics": {"warnings": []}}
    apply_baseline_drift_warning(summary, True, "baseline drift detected")
    assert summary["diagnostics"]["baseline_compat_warning"] is True
    assert "baseline drift detected" in summary["diagnostics"]["warnings"]

    # When no warning, the flag should be cleared but no message added
    apply_baseline_drift_warning(summary, False, None)
    assert summary["diagnostics"]["baseline_compat_warning"] is False
