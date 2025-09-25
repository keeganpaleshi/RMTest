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
    assert abs(est - A_true) <= sigma


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


def test_single_isotope_po218_zero_counts_returns_zero_with_inf_uncertainty():
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
    assert math.isinf(result["stat_unc_Bq"])

    comp218 = result["components"]["from_po218"]
    assert comp218["counts"] == 0
    assert comp218["activity_Bq"] == pytest.approx(0.0)
    assert math.isinf(comp218["variance"])


def test_single_isotope_po214_zero_counts_returns_zero_with_inf_uncertainty():
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
    assert math.isinf(result["stat_unc_Bq"])

    comp214 = result["components"]["from_po214"]
    assert comp214["counts"] == 0
    assert comp214["activity_Bq"] == pytest.approx(0.0)
    assert math.isinf(comp214["variance"])

