"""Estimate radon activity from Po-218 and Po-214 counts."""

from __future__ import annotations

import math
from typing import Any, Mapping

from constants import load_nuclide_overrides, RN222, PO218, PO214
from radon_activity import compute_radon_activity

__all__ = ["estimate_radon_activity"]

# One-sided 95% upper limit on the Poisson mean for zero observed counts.
UL95_POISSON_MEAN = -math.log(0.05)


def estimate_radon_activity(
    N218: int | None = None,
    epsilon218: float | None = None,
    f218: float | None = None,
    N214: int | None = None,
    epsilon214: float | None = None,
    f214: float | None = None,
    *,
    live_time218_s: float | None = None,
    live_time214_s: float | None = None,
    rate214: float | None = None,
    err214: float | None = None,
    rate218: float | None = None,
    err218: float | None = None,
    analysis_isotope: str = "radon",
    joint_equilibrium: bool = False,
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
    live_time218_s, live_time214_s : float, optional
        Measurement live time in seconds corresponding to the Po-218/Po-214
        counts. Required whenever the respective ``N`` value is provided.
    analysis_isotope : {"radon", "po218", "po214"}
        Determines whether to combine the estimates or return a single isotope.
    joint_equilibrium : bool, optional
        When ``True`` and both Po-218 and Po-214 counts are provided, fit a
        single radon activity parameter shared by both isotopes instead of
        independently estimating each daughter.  This enforces physical
        consistency under the assumption of secular equilibrium.  Defaults to
        ``False``.
    nuclide_constants : mapping, optional
        Overrides for nuclide half-lives.  Must contain ``Rn222``, ``Po218`` and
        ``Po214`` entries compatible with :mod:`constants`.

    Notes
    -----
    When either isotope records zero counts, the returned variance is set to
    ``math.nan`` to explicitly signal that a Gaussian approximation to the
    uncertainty is not valid in that regime.  Components include a
    ``gaussian_uncertainty_valid`` flag to make this behavior explicit.
    """
    if rate214 is not None or rate218 is not None:
        comp: dict[str, Any] = {}
        if rate218 is not None:
            comp["from_po218"] = {
                "activity_Bq": rate218,
                "stat_unc_Bq": err218,
            }
        if rate214 is not None:
            comp["from_po214"] = {
                "activity_Bq": rate214,
                "stat_unc_Bq": err214,
            }
        A, sigma = compute_radon_activity(
            rate218,
            err218,
            1.0,
            rate214,
            err214,
            1.0,
            require_equilibrium=False,
        )
        mode = analysis_isotope.lower()
        if mode not in {"radon", "po214", "po218"}:
            raise ValueError("invalid isotope mode")
        if mode == "radon":
            return {
                "isotope_mode": "radon",
                "Rn_activity_Bq": A,
                "stat_unc_Bq": sigma,
                "components": comp,
            }
        if mode == "po214":
            if rate214 is None:
                raise ValueError(
                    "Po-214 rate unavailable for requested analysis_isotope='po214'"
                )
            return {
                "isotope_mode": "po214",
                "Rn_activity_Bq": rate214,
                "activity_Bq": rate214,
                "stat_unc_Bq": err214,
                "components": comp,
            }
        if rate218 is None:
            raise ValueError(
                "Po-218 rate unavailable for requested analysis_isotope='po218'"
            )
        return {
            "isotope_mode": "po218",
            "Rn_activity_Bq": rate218,
            "activity_Bq": rate218,
            "stat_unc_Bq": err218,
            "components": comp,
        }

    has218 = N218 is not None
    has214 = N214 is not None

    if has218 and (epsilon218 is None or f218 is None):
        raise ValueError("UL95 computation requires efficiencies and fractions")
    if has214 and (epsilon214 is None or f214 is None):
        raise ValueError("UL95 computation requires efficiencies and fractions")

    if has218 and epsilon218 <= 0:
        raise ValueError("efficiencies must be positive")
    if has214 and epsilon214 <= 0:
        raise ValueError("efficiencies must be positive")
    if has218 and f218 <= 0:
        raise ValueError("fractions must be positive")
    if has214 and f214 <= 0:
        raise ValueError("fractions must be positive")

    # Handle the special case when both counts are zero.  This signals that the
    # activity is unconstrained under a Gaussian approximation and is better
    # represented as an upper limit than as a Gaussian sigma.
    if (N218 or 0) + (N214 or 0) == 0:
        components: dict[str, Any] = {}
        note = "joint pooled estimator" if joint_equilibrium and has218 and has214 else None
        if N218 is not None:
            components["from_po218"] = {
                "counts": N218,
                "activity_Bq": 0.0,
                "variance": math.nan,
                "gaussian_uncertainty_valid": False,
            }
            if note:
                components["from_po218"]["note"] = note
        if N214 is not None:
            components["from_po214"] = {
                "counts": N214,
                "activity_Bq": 0.0,
                "variance": math.nan,
                "gaussian_uncertainty_valid": False,
            }
            if note:
                components["from_po214"]["note"] = note

        coeff_sum = 0.0
        if has218 and live_time218_s and live_time218_s > 0:
            if epsilon218 is None or f218 is None:
                raise ValueError("UL95 computation requires efficiencies and fractions")
            coeff_sum += epsilon218 * f218 * live_time218_s
        if has214 and live_time214_s and live_time214_s > 0:
            if epsilon214 is None or f214 is None:
                raise ValueError("UL95 computation requires efficiencies and fractions")
            coeff_sum += epsilon214 * f214 * live_time214_s
        rn_ul95 = UL95_POISSON_MEAN / coeff_sum if coeff_sum > 0 else None

        result: dict[str, Any] = {
            "isotope_mode": analysis_isotope.lower(),
            "Rn_activity_Bq": 0.0,
            "stat_unc_Bq": math.nan,
            "gaussian_uncertainty_valid": False,
            "components": components,
            "joint_equilibrium": joint_equilibrium,
        }
        if rn_ul95 is not None:
            result["Rn_activity_UL95_Bq"] = rn_ul95
        return result

    consts = load_nuclide_overrides(nuclide_constants)
    # Accessing the constants keeps API compatibility for callers that expect
    # half-life overrides to be validated, even though they no longer enter the
    # counts-based rate calculation.
    _ = consts.get("Rn222", RN222)
    _ = consts.get("Po218", PO218)
    _ = consts.get("Po214", PO214)

    def _estimate(
        counts: int | None,
        eff: float | None,
        frac: float | None,
        live_time: float | None,
        label: str,
    ) -> tuple[float, float, bool, float | None] | None:
        if counts is None:
            return None
        if eff is None or frac is None:
            raise ValueError("UL95 computation requires efficiencies and fractions")
        if counts < 0:
            raise ValueError(f"counts for {label} must be non-negative")
        if counts == 0:
            if live_time is not None and live_time < 0:
                raise ValueError(f"live_time for {label} must be non-negative")
            coeff = eff * frac * live_time if live_time and live_time > 0 else None
            ul95 = UL95_POISSON_MEAN / coeff if coeff else None
            return 0.0, math.nan, False, ul95
        if live_time is None or live_time <= 0:
            raise ValueError(f"live_time for {label} must be positive")
        rn = counts / (eff * frac * live_time)
        var = counts / (eff**2 * frac**2 * live_time**2)
        return rn, var, True, None

    res218 = _estimate(N218, epsilon218, f218, live_time218_s, "Po-218")
    res214 = _estimate(N214, epsilon214, f214, live_time214_s, "Po-214")

    components: dict[str, Any] = {}
    if res218:
        rn218, var218, gaussian_ok218, ul95_218 = res218
        components["from_po218"] = {
            "counts": N218,
            "activity_Bq": rn218,
            "variance": var218,
            "gaussian_uncertainty_valid": gaussian_ok218,
        }
        if ul95_218 is not None:
            components["from_po218"]["Rn_activity_UL95_Bq"] = ul95_218
    if res214:
        rn214, var214, gaussian_ok214, ul95_214 = res214
        components["from_po214"] = {
            "counts": N214,
            "activity_Bq": rn214,
            "variance": var214,
            "gaussian_uncertainty_valid": gaussian_ok214,
        }
        if ul95_214 is not None:
            components["from_po214"]["Rn_activity_UL95_Bq"] = ul95_214

    mode = analysis_isotope.lower()
    if mode not in {"radon", "po218", "po214"}:
        raise ValueError("invalid isotope mode")

    if mode == "po218":
        if not res218:
            raise ValueError("Po-218 counts unavailable for requested analysis_isotope='po218'")
        rn, var, gaussian_valid, ul95 = res218
        result = {
            "isotope_mode": "po218",
            "Rn_activity_Bq": rn,
            "stat_unc_Bq": math.sqrt(var) if not math.isnan(var) else math.nan,
            "gaussian_uncertainty_valid": gaussian_valid and math.isfinite(var),
            "components": components,
        }
        if ul95 is not None:
            result["Rn_activity_UL95_Bq"] = ul95
        return result
    if mode == "po214":
        if not res214:
            raise ValueError("Po-214 counts unavailable for requested analysis_isotope='po214'")
        rn, var, gaussian_valid, ul95 = res214
        result = {
            "isotope_mode": "po214",
            "Rn_activity_Bq": rn,
            "stat_unc_Bq": math.sqrt(var) if not math.isnan(var) else math.nan,
            "gaussian_uncertainty_valid": gaussian_valid and math.isfinite(var),
            "components": components,
        }
        if ul95 is not None:
            result["Rn_activity_UL95_Bq"] = ul95
        return result

    # For joint equilibrium, both live times must be valid (not None and positive)
    if (
        joint_equilibrium
        and res218
        and res214
        and mode == "radon"
        and live_time218_s is not None
        and live_time218_s > 0
        and live_time214_s is not None
        and live_time214_s > 0
    ):
        coeff218 = epsilon218 * f218 * live_time218_s
        coeff214 = epsilon214 * f214 * live_time214_s
        coeff_sum = coeff218 + coeff214
        counts_sum = (N218 or 0) + (N214 or 0)
        if coeff_sum <= 0:
            raise ValueError("live_time and efficiency products must be positive")
        rn = counts_sum / coeff_sum
        if counts_sum == 0:
            var = math.nan
            gaussian_valid = False
            rn_ul95 = UL95_POISSON_MEAN / coeff_sum
        else:
            var = counts_sum / (coeff_sum**2)
            gaussian_valid = True
            rn_ul95 = None

        components["from_po218"] = {
            "counts": N218,
            "activity_Bq": rn,
            "variance": var,
            "gaussian_uncertainty_valid": gaussian_valid,
            "note": "joint pooled estimator",
        }
        components["from_po214"] = {
            "counts": N214,
            "activity_Bq": rn,
            "variance": var,
            "gaussian_uncertainty_valid": gaussian_valid,
            "note": "joint pooled estimator",
        }

        result = {
            "isotope_mode": "radon",
            "Rn_activity_Bq": rn,
            "stat_unc_Bq": math.sqrt(var) if math.isfinite(var) else math.nan,
            "gaussian_uncertainty_valid": gaussian_valid and math.isfinite(var),
            "components": components,
            "joint_equilibrium": True,
        }
        if rn_ul95 is not None:
            result["Rn_activity_UL95_Bq"] = rn_ul95
        return result

    # Combine both when possible
    if res218 and res214:
        rn218, var218, gaussian_ok218, _ = res218
        rn214, var214, gaussian_ok214, _ = res214
        w218 = 1.0 / var218 if math.isfinite(var218) else 0.0
        w214 = 1.0 / var214 if math.isfinite(var214) else 0.0
        denom = w218 + w214
        if denom > 0:
            rn_comb = (w218 * rn218 + w214 * rn214) / denom
            sigma = 1.0 / math.sqrt(denom)
            rn = rn_comb
            var = sigma**2
            gaussian_valid = (
                (gaussian_ok218 and math.isfinite(var218))
                or (gaussian_ok214 and math.isfinite(var214))
            )
        else:
            rn = 0.0
            var = math.nan
            gaussian_valid = False
    elif res218:
        rn, var, gaussian_valid, _ = res218
    elif res214:
        rn, var, gaussian_valid, _ = res214
    else:
        rn = 0.0
        var = math.nan
        gaussian_valid = False

    return {
        "isotope_mode": "radon",
        "Rn_activity_Bq": rn,
        "stat_unc_Bq": math.sqrt(var) if not math.isnan(var) else math.nan,
        "gaussian_uncertainty_valid": gaussian_valid and math.isfinite(var),
        "components": components,
    }
