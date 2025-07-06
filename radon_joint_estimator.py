"""Joint radon activity estimator using Po-218 and Po-214 counts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from constants import PO214, PO218, RN222


@dataclass
class Result:
    isotope_mode: str
    Rn_activity_Bq: float
    stat_unc_Bq: float
    components: dict[str, dict[str, float]]


def estimate(
    N218: float | None,
    eff218: float = 1.0,
    f218: float = 1.0,
    N214: float | None = None,
    eff214: float = 1.0,
    f214: float = 1.0,
    *,
    analysis_isotope: str = "radon",
) -> Result:
    """Return radon activity estimate based on Po-218/Po-214 counts."""

    lam_rn = math.log(2.0) / RN222.half_life_s
    lam_218 = math.log(2.0) / PO218.half_life_s
    lam_214 = math.log(2.0) / PO214.half_life_s

    out_components: dict[str, dict[str, float]] = {}

    Rn_from_218 = None
    var218 = None
    if N218 is not None and N218 > 0 and eff218 > 0 and f218 > 0:
        Rn_from_218 = N218 / (eff218 * f218) * (lam_rn / lam_218)
        var218 = Rn_from_218 / N218
        out_components["from_po218"] = {
            "counts": float(N218),
            "estimate_Bq": float(Rn_from_218),
            "variance": float(var218),
        }

    Rn_from_214 = None
    var214 = None
    if N214 is not None and N214 > 0 and eff214 > 0 and f214 > 0:
        Rn_from_214 = N214 / (eff214 * f214) * (lam_rn / lam_214)
        var214 = Rn_from_214 / N214
        out_components["from_po214"] = {
            "counts": float(N214),
            "estimate_Bq": float(Rn_from_214),
            "variance": float(var214),
        }

    mode = analysis_isotope.lower()

    if mode == "po218" and Rn_from_218 is not None:
        sigma = math.sqrt(var218)
        return Result("po218", float(Rn_from_218), sigma, out_components)
    if mode == "po214" and Rn_from_214 is not None:
        sigma = math.sqrt(var214)
        return Result("po214", float(Rn_from_214), sigma, out_components)

    if Rn_from_218 is None and Rn_from_214 is None:
        return Result("radon", 0.0, math.nan, out_components)

    if Rn_from_218 is None:
        sigma = math.sqrt(var214)
        return Result("radon", float(Rn_from_214), sigma, out_components)
    if Rn_from_214 is None:
        sigma = math.sqrt(var218)
        return Result("radon", float(Rn_from_218), sigma, out_components)

    w218 = 1.0 / var218
    w214 = 1.0 / var214
    Rn_comb = (w218 * Rn_from_218 + w214 * Rn_from_214) / (w218 + w214)
    sigma = 1.0 / math.sqrt(w218 + w214)

    return Result("radon", float(Rn_comb), float(sigma), out_components)
