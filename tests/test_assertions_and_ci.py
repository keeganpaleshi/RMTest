import pytest
from assertions_and_ci import run_assertions


def test_run_assertions_ok():
    summary = {
        "baseline": {"corrected_activity": {"Po214": {"value": 0.1}}},
        "radon": {"Rn_activity_Bq": 10.0, "stat_unc_Bq": 1.0},
    }
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    run_assertions(summary, constants, config)


def test_run_assertions_negative_activity():
    summary = {"baseline": {"corrected_activity": {"Po214": {"value": -1.0}}}}
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    with pytest.raises(AssertionError):
        run_assertions(summary, constants, config)


def test_run_assertions_invalid_radon_activity():
    summary = {
        "baseline": {"corrected_activity": {"Po214": {"value": 0.1}}},
        "radon": {"Rn_activity_Bq": -1.0, "stat_unc_Bq": 1.0},
    }
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    with pytest.raises(AssertionError):
        run_assertions(summary, constants, config)


def test_run_assertions_invalid_half_life():
    summary = {"baseline": {"corrected_activity": {"Po214": {"value": 0.1}}}}
    constants = {"Po214": {"half_life_s": 2000.0}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    with pytest.raises(AssertionError):
        run_assertions(summary, constants, config)


def test_run_assertions_invalid_radon_uncertainty():
    summary = {
        "baseline": {"corrected_activity": {"Po214": {"value": 0.1}}},
        "radon": {"Rn_activity_Bq": 1.0, "stat_unc_Bq": 0.0},
    }
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    with pytest.raises(AssertionError):
        run_assertions(summary, constants, config)


def test_run_assertions_invalid_sample_volume():
    summary = {"baseline": {"corrected_activity": {"Po214": {"value": 0.1}}}}
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 0.0}}
    with pytest.raises(AssertionError):
        run_assertions(summary, constants, config)
