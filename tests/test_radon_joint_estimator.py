import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radon_joint_estimator import estimate_radon_activity
from constants import load_nuclide_overrides


def test_radon_joint_estimator_combination():
    consts = load_nuclide_overrides(None)
    assert consts["Rn222"].half_life_s > 0

    A_true = 0.0063
    eff218 = 0.8
    eff214 = 0.7
    f218 = 1.0
    f214 = 1e-7

    live_time218 = 7200.0
    live_time214 = 7200.0

    N218 = round(A_true * eff218 * f218 * live_time218)
    N214 = round(A_true * eff214 * f214 * live_time214)

    result = estimate_radon_activity(
        N218,
        eff218,
        f218,
        N214,
        eff214,
        f214,
        live_time218_s=live_time218,
        live_time214_s=live_time214,
    )

    assert result["isotope_mode"] == "radon"
    est = result["Rn_activity_Bq"]
    sigma = result["stat_unc_Bq"]
    assert result["gaussian_uncertainty_valid"] is True
    assert abs(est - A_true) <= sigma


def test_radon_joint_estimator_joint_equilibrium_fit():
    A_true = 0.012
    eff218 = 0.9
    eff214 = 0.8
    f218 = 1.0
    f214 = 0.5

    live_time218 = 3600.0
    live_time214 = 5400.0

    coeff_sum = eff218 * f218 * live_time218 + eff214 * f214 * live_time214
    counts_total = round(A_true * coeff_sum)

    # Split counts unevenly to ensure the joint fit really ties the isotopes
    N218 = round(counts_total * 0.65)
    N214 = counts_total - N218

    result = estimate_radon_activity(
        N218,
        eff218,
        f218,
        N214,
        eff214,
        f214,
        live_time218_s=live_time218,
        live_time214_s=live_time214,
        joint_equilibrium=True,
    )

    assert result["isotope_mode"] == "radon"
    assert result.get("joint_equilibrium") is True
    assert result["gaussian_uncertainty_valid"] is True

    est = result["Rn_activity_Bq"]
    sigma = result["stat_unc_Bq"]
    assert abs(est - A_true) <= 2 * sigma

    comp218 = result["components"]["from_po218"]
    comp214 = result["components"]["from_po214"]
    assert comp218["activity_Bq"] == pytest.approx(est)
    assert comp214["activity_Bq"] == pytest.approx(est)

    expected_variance = counts_total / (coeff_sum**2)
    assert comp218["variance"] == pytest.approx(expected_variance)
    assert comp214["variance"] == pytest.approx(expected_variance)


def test_counts_without_live_time_raises():
    with pytest.raises(ValueError):
        estimate_radon_activity(
            N218=10,
            epsilon218=0.5,
            f218=1.0,
            N214=None,
            epsilon214=1.0,
            f214=1.0,
        )


def test_single_isotope_po218_only_requires_matching_metadata():
    result = estimate_radon_activity(
        N218=42,
        epsilon218=0.5,
        f218=0.9,
        live_time218_s=1800.0,
        analysis_isotope="po218",
    )

    assert result["isotope_mode"] == "po218"
    assert "from_po218" in result["components"]
    assert "from_po214" not in result["components"]
    assert result["gaussian_uncertainty_valid"] is True


def test_single_isotope_po214_only_requires_matching_metadata():
    result = estimate_radon_activity(
        N214=13,
        epsilon214=0.6,
        f214=0.8,
        live_time214_s=2400.0,
        analysis_isotope="po214",
    )

    assert result["isotope_mode"] == "po214"
    assert "from_po214" in result["components"]
    assert "from_po218" not in result["components"]
    assert result["gaussian_uncertainty_valid"] is True


def test_single_isotope_po218_missing_counts_raises():
    with pytest.raises(ValueError, match="Po-218 counts unavailable"):
        estimate_radon_activity(
            N218=None,
            epsilon218=0.5,
            f218=1.0,
            N214=20,
            epsilon214=0.6,
            f214=1.0,
            live_time214_s=3600.0,
            analysis_isotope="po218",
        )


def test_single_isotope_po214_missing_counts_raises():
    with pytest.raises(ValueError, match="Po-214 counts unavailable"):
        estimate_radon_activity(
            N218=15,
            epsilon218=0.5,
            f218=1.0,
            N214=None,
            epsilon214=0.6,
            f214=1.0,
            live_time218_s=3600.0,
            analysis_isotope="po214",
        )


def test_rate_shortcut_requires_requested_isotope_data():
    with pytest.raises(ValueError, match="Po-214 rate unavailable"):
        estimate_radon_activity(
            rate218=1.0,
            err218=0.1,
            analysis_isotope="po214",
        )

    with pytest.raises(ValueError, match="Po-218 rate unavailable"):
        estimate_radon_activity(
            rate214=0.9,
            err214=0.05,
            analysis_isotope="po218",
        )


def test_rate_shortcut_single_isotope_preserves_rn_key():
    res_po218 = estimate_radon_activity(
        rate218=0.42,
        err218=0.07,
        analysis_isotope="po218",
    )

    assert res_po218["isotope_mode"] == "po218"
    assert res_po218["Rn_activity_Bq"] == pytest.approx(0.42)
    assert res_po218["activity_Bq"] == pytest.approx(0.42)

    res_po214 = estimate_radon_activity(
        rate214=0.31,
        err214=0.05,
        analysis_isotope="po214",
    )

    assert res_po214["isotope_mode"] == "po214"
    assert res_po214["Rn_activity_Bq"] == pytest.approx(0.31)
    assert res_po214["activity_Bq"] == pytest.approx(0.31)


def test_single_isotope_po218_zero_counts_returns_zero_with_nan_uncertainty():
    result = estimate_radon_activity(
        N218=0,
        epsilon218=0.5,
        f218=1.0,
        N214=25,
        epsilon214=0.6,
        f214=1.0,
        live_time218_s=3600.0,
        live_time214_s=3600.0,
        analysis_isotope="po218",
    )

    assert result["isotope_mode"] == "po218"
    assert result["Rn_activity_Bq"] == pytest.approx(0.0)
    assert math.isnan(result["stat_unc_Bq"])
    assert result["gaussian_uncertainty_valid"] is False
    assert "Rn_activity_UL95_Bq" in result
    assert result["Rn_activity_UL95_Bq"] == pytest.approx(3.0 / (0.5 * 1.0 * 3600.0))

    comp218 = result["components"]["from_po218"]
    assert comp218["counts"] == 0
    assert comp218["activity_Bq"] == pytest.approx(0.0)
    assert math.isnan(comp218["variance"])
    assert comp218["gaussian_uncertainty_valid"] is False


def test_single_isotope_po214_zero_counts_returns_zero_with_nan_uncertainty():
    result = estimate_radon_activity(
        N218=30,
        epsilon218=0.5,
        f218=1.0,
        N214=0,
        epsilon214=0.6,
        f214=1.0,
        live_time218_s=3600.0,
        live_time214_s=3600.0,
        analysis_isotope="po214",
    )

    assert result["isotope_mode"] == "po214"
    assert result["Rn_activity_Bq"] == pytest.approx(0.0)
    assert math.isnan(result["stat_unc_Bq"])
    assert result["gaussian_uncertainty_valid"] is False
    assert "Rn_activity_UL95_Bq" in result
    assert result["Rn_activity_UL95_Bq"] == pytest.approx(3.0 / (0.6 * 1.0 * 3600.0))

    comp214 = result["components"]["from_po214"]
    assert comp214["counts"] == 0
    assert comp214["activity_Bq"] == pytest.approx(0.0)
    assert math.isnan(comp214["variance"])
    assert comp214["gaussian_uncertainty_valid"] is False


def test_joint_equilibrium_zero_counts_returns_nan_uncertainty_and_note():
    result = estimate_radon_activity(
        N218=0,
        epsilon218=0.5,
        f218=1.0,
        N214=0,
        epsilon214=0.6,
        f214=1.0,
        live_time218_s=3600.0,
        live_time214_s=3600.0,
        joint_equilibrium=True,
    )

    assert result["isotope_mode"] == "radon"
    assert math.isnan(result["stat_unc_Bq"])
    assert result["gaussian_uncertainty_valid"] is False
    assert result["Rn_activity_UL95_Bq"] == pytest.approx(3.0 / ((0.5 + 0.6) * 3600.0))

    comp218 = result["components"]["from_po218"]
    comp214 = result["components"]["from_po214"]
    assert math.isnan(comp218["variance"])
    assert math.isnan(comp214["variance"])
    assert comp218["note"] == "joint pooled estimator"
    assert comp214["note"] == "joint pooled estimator"

