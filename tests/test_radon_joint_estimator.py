import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radon_joint_estimator import estimate_radon_activity
from constants import load_nuclide_overrides


def test_radon_joint_estimator_combination():
    consts = load_nuclide_overrides(None)
    lam_rn = math.log(2.0) / consts["Rn222"].half_life_s
    lam_218 = math.log(2.0) / consts["Po218"].half_life_s
    lam_214 = math.log(2.0) / consts["Po214"].half_life_s

    A_true = 0.0063
    eff218 = 0.8
    eff214 = 0.7
    f218 = 1.0
    f214 = 1e-7

    N218 = round(A_true * eff218 * f218 * lam_218 / lam_rn)
    N214 = round(A_true * eff214 * f214 * lam_214 / lam_rn)

    result = estimate_radon_activity(
        N218,
        eff218,
        f218,
        N214,
        eff214,
        f214,
    )

    assert result["isotope_mode"] == "radon"
    est = result["Rn_activity_Bq"]
    sigma = result["stat_unc_Bq"]
    assert abs(est - A_true) <= sigma


def test_radon_joint_estimator_zero_counts():
    result = estimate_radon_activity(0, 1.0, 1.0, 0, 1.0, 1.0)
    assert result["Rn_activity_Bq"] == 0.0
    assert math.isinf(result["stat_unc_Bq"])

