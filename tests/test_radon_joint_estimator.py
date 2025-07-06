import sys
from pathlib import Path
import math
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radon_joint_estimator import estimate
from constants import RN222, PO218, PO214


def test_joint_estimator_unbiased():
    lam_rn = math.log(2.0) / RN222.half_life_s
    lam_218 = math.log(2.0) / PO218.half_life_s
    lam_214 = math.log(2.0) / PO214.half_life_s

    R_true = 0.012
    eff218 = 0.6
    eff214 = 0.7
    f218 = 1.0
    f214 = 1.0

    N218 = R_true * eff218 * f218 * (lam_218 / lam_rn)
    N214 = R_true * eff214 * f214 * (lam_214 / lam_rn)

    res = estimate(N218, eff218, f218, N214, eff214, f214)
    assert res.isotope_mode == "radon"
    assert abs(res.Rn_activity_Bq - R_true) <= res.stat_unc_Bq
