import pytest

from baseline_handling import (
    annotate_summary_with_baseline,
    build_baseline_record,
    get_fixed_background_for_time_fit,
    record_baseline_drift_warning,
    set_intrinsic_activity,
    set_scale,
)


def test_fixed_background_uses_scale():
    record = build_baseline_record(
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T01:00:00Z",
        live_time_s=3600,
    )
    set_intrinsic_activity(record, "Po214", 2.0, 0.4)
    set_scale(record, "Po214", 0.25)

    fixed = get_fixed_background_for_time_fit(record, "Po214", {})
    assert fixed.mode == "baseline_fixed"
    assert fixed.rate_Bq == pytest.approx(0.5)
    assert fixed.uncertainty_Bq == pytest.approx(0.1)


def test_fixed_background_falls_back_to_config_dilution():
    record = build_baseline_record(
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T01:00:00Z",
        live_time_s=3600,
    )
    set_intrinsic_activity(record, "Po214", 1.0, 0.2)
    config = {"baseline": {"monitor_volume_l": 1.0, "sample_volume_l": 1.0}}

    fixed = get_fixed_background_for_time_fit(record, "Po214", config)
    assert fixed.rate_Bq == pytest.approx(0.5)
    assert fixed.uncertainty_Bq == pytest.approx(0.1)


def test_annotate_summary_with_baseline_provenance():
    record = build_baseline_record(
        start="2024-01-01T00:00:00Z",
        end="2024-01-02T00:00:00Z",
    )
    summary = {
        "baseline": {"start": "2024-01-01T00:00:00Z"},
        "time_fit": {"Po214": {"E_Po214": 1.0}},
        "radon": {},
    }
    meta = {
        "Po214": {
            "mode": "baseline_fixed",
            "baseline_rate_Bq": 0.2,
            "baseline_unc_Bq": 0.05,
        }
    }

    annotate_summary_with_baseline(summary, record, meta)

    tf = summary["time_fit"]["Po214"]
    assert tf["background_source"] == "baseline_fixed"
    assert tf["baseline_activity_Bq"] == pytest.approx(0.2)
    assert tf["baseline_activity_unc_Bq"] == pytest.approx(0.05)
    assert tf["baseline_source_range"] == record["timestamp_range"]
    assert summary["radon"]["background_source"] == "baseline_fixed"


def test_record_baseline_drift_warning_sets_flag():
    record = build_baseline_record()
    record["po214_centroid_mev"] = 7.68
    record["po214_sigma_E_mev"] = 0.2
    calibration = {"peaks": {"Po214": {"centroid_mev": 7.78}}, "sigma_E": 0.25}
    summary = {"diagnostics": {}}

    record_baseline_drift_warning(
        summary,
        record,
        calibration,
        energy_tol_mev=0.05,
        sigma_tol_fraction=0.1,
    )

    assert summary["diagnostics"]["baseline_compat_warning"] is True
    assert "baseline_compat_message" in summary["diagnostics"]
