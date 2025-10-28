import pytest

import baseline_handling


def test_get_fixed_background_applies_scale():
    record = {
        "rates_Bq": {"Po214": 0.5},
        "rate_unc_Bq": {"Po214": 0.1},
        "scale_factors": {"Po214": 0.2},
        "timestamp_range": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        "dilution_factor": 0.2,
    }

    result = baseline_handling.get_fixed_background_for_time_fit(
        record,
        "Po214",
        {},
    )

    assert result is not None
    assert result["background_rate_Bq"] == pytest.approx(0.1)
    assert result["background_unc_Bq"] == pytest.approx(0.02)
    assert result["source_range"] == record["timestamp_range"]
    assert result["dilution_factor"] == record["dilution_factor"]


def test_assess_baseline_drift_flags_large_delta():
    record = {
        "po214_centroid_MeV": 7.6,
        "po214_sigma_E_MeV": 0.1,
    }
    calibration = {
        "sigma_E": 0.15,
        "peaks": {"Po214": {"centroid_energy_MeV": 7.7}},
    }

    flag, message = baseline_handling.assess_baseline_drift(record, calibration)

    assert flag is True
    assert message
