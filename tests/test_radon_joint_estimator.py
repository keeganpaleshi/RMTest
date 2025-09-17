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

