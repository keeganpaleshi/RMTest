"""Estimate radon activity from Po-218 and Po-214 counts."""

from __future__ import annotations

import math
from typing import Any, Mapping

from constants import load_nuclide_overrides, RN222, PO218, PO214

__all__ = ["estimate_radon_activity"]


def _decay_constant(hl_s: float) -> float:
    if hl_s <= 0:
        raise ValueError("half-life must be positive")
    return math.log(2.0) / hl_s


def estimate_radon_activity(
    N218: int | None,
    epsilon218: float,
    f218: float,
    N214: int | None,
    epsilon214: float,
    f214: float,
    *,
    analysis_isotope: str = "radon",
    nuclide_constants: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate radon activity from Po-218/Po-214 counts.

    Parameters
    ----------
    N218, N214 : int or None
        Observed counts for Po-218 and Po-214.
    epsilon218, epsilon214 : float
        Detection efficiencies for each isotope.
    f218, f214 : float
        Fraction of progeny decaying inside the cell.
    analysis_isotope : {"radon", "po218", "po214"}
        Determines whether to combine the estimates or return a single isotope.
    nuclide_constants : mapping, optional
        Overrides for nuclide half-lives.  Must contain ``Rn222``, ``Po218`` and
        ``Po214`` entries compatible with :mod:`constants`.
    """
    if epsilon218 <= 0 or epsilon214 <= 0:
        raise ValueError("efficiencies must be positive")
    if f218 <= 0 or f214 <= 0:
        raise ValueError("fractions must be positive")

    consts = load_nuclide_overrides(nuclide_constants)
    lam_rn = _decay_constant(consts.get("Rn222", RN222).half_life_s)
    lam_218 = _decay_constant(consts.get("Po218", PO218).half_life_s)
    lam_214 = _decay_constant(consts.get("Po214", PO214).half_life_s)

    def _estimate(counts: int | None, eff: float, frac: float, lam_dau: float):
        if counts is None or counts <= 0:
            return None
        rn = counts / (eff * frac) * (lam_rn / lam_dau)
        var = rn / counts
        return rn, var

    res218 = _estimate(N218, epsilon218, f218, lam_218)
    res214 = _estimate(N214, epsilon214, f214, lam_214)

    components: dict[str, Any] = {}
    if res218:
        rn218, var218 = res218
        components["from_po218"] = {
            "counts": N218,
            "activity_Bq": rn218,
            "variance": var218,
        }
    if res214:
        rn214, var214 = res214
        components["from_po214"] = {
            "counts": N214,
            "activity_Bq": rn214,
            "variance": var214,
        }

    mode = analysis_isotope.lower()
    if mode not in {"radon", "po218", "po214"}:
        raise ValueError("invalid isotope mode")

    if mode == "po218" and res218:
        rn, var = res218
        return {
            "isotope_mode": "po218",
            "Rn_activity_Bq": rn,
            "stat_unc_Bq": math.sqrt(var),
            "components": components,
        }
    if mode == "po214" and res214:
        rn, var = res214
        return {
            "isotope_mode": "po214",
            "Rn_activity_Bq": rn,
            "stat_unc_Bq": math.sqrt(var),
            "components": components,
        }

    # Combine both when possible
    if res218 and res214:
        rn218, var218 = res218
        rn214, var214 = res214
        w218 = 1.0 / var218
        w214 = 1.0 / var214
        rn_comb = (w218 * rn218 + w214 * rn214) / (w218 + w214)
        sigma = 1.0 / math.sqrt(w218 + w214)
        rn = rn_comb
        var = sigma ** 2
    elif res218:
        rn, var = res218
    elif res214:
        rn, var = res214
    else:
        rn = 0.0
        var = math.nan

    return {
        "isotope_mode": "radon",
        "Rn_activity_Bq": rn,
        "stat_unc_Bq": math.sqrt(var) if not math.isnan(var) else math.nan,
        "components": components,
    }
