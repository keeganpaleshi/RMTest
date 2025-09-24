import pytest

from assertions_and_ci import run_assertions
from constants import load_nuclide_overrides


def test_run_assertions_ok():
    summary = {"radon": {"Rn_activity_Bq": 1.0, "stat_unc_Bq": 0.1}}
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    run_assertions(summary, constants, config)


def test_run_assertions_negative_activity():
    summary = {"radon": {"Rn_activity_Bq": -1.0, "stat_unc_Bq": 0.1}}
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    with pytest.raises(AssertionError):
        run_assertions(summary, constants, config)


def test_run_assertions_invalid_half_life():
    summary = {"radon": {"Rn_activity_Bq": 1.0, "stat_unc_Bq": 0.1}}
    constants = {"Po214": {"half_life_s": 2000.0}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    with pytest.raises(AssertionError):
        run_assertions(summary, constants, config)


def test_run_assertions_zero_sample_volume_allowed():
    summary = {"radon": {"Rn_activity_Bq": 1.0, "stat_unc_Bq": 0.1}}
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 0.0}}
    run_assertions(summary, constants, config)


def test_run_assertions_zero_uncertainty_allowed():
    summary = {"radon": {"Rn_activity_Bq": 1.0, "stat_unc_Bq": 0.0}}
    constants = {"Po214": {"half_life_s": 0.000164}}
    config = {"baseline": {"sample_volume_l": 1.0}}
    run_assertions(summary, constants, config)


def test_run_assertions_accepts_dataclass_constants():
    summary = {"radon": {"Rn_activity_Bq": 1.0, "stat_unc_Bq": 0.1}}
    constants = load_nuclide_overrides({})
    config = {"baseline": {"sample_volume_l": 1.0}}
    run_assertions(summary, constants, config)
