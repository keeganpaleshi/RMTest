"""DNL cross-validation: test whether the self-estimated DNL correction
captures real ADC hardware structure or absorbs model misfit.

The test splits the event list into two independent time halves, estimates
DNL factors from each half separately, and then cross-applies them:

- **Self-DNL**: DNL estimated and applied to the same half (the normal pipeline)
- **Cross-DNL**: DNL estimated from one half, applied to the other
- **No-DNL**: Fit without any DNL correction

If cross-DNL improves held-out NLL relative to no-DNL, the correction
captures a real, time-stable hardware feature (ADC channel-width
non-uniformity).  If self-DNL >> cross-DNL improvement, the excess is
model-misfit absorption.
"""

from __future__ import annotations

import copy
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DNLCrossValResult:
    """Results of DNL cross-validation."""

    # NLL values for each half under each condition
    nll_no_dnl_A: float = np.nan
    nll_self_dnl_A: float = np.nan
    nll_cross_dnl_A: float = np.nan
    nll_no_dnl_B: float = np.nan
    nll_self_dnl_B: float = np.nan
    nll_cross_dnl_B: float = np.nan

    # DNL factors from each half
    dnl_factors_A: np.ndarray | None = None
    dnl_factors_B: np.ndarray | None = None

    # Derived metrics
    cross_improvement_A: float = np.nan  # nll_no_dnl_A - nll_cross_dnl_A
    cross_improvement_B: float = np.nan
    self_improvement_A: float = np.nan   # nll_no_dnl_A - nll_self_dnl_A
    self_improvement_B: float = np.nan
    overfitting_indicator_A: float = np.nan  # self - cross improvement
    overfitting_indicator_B: float = np.nan
    dnl_correlation: float = np.nan      # Pearson corr of DNL factors

    verdict: str = "unknown"
    verdict_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary for JSON output."""
        d = {
            "nll_no_dnl_A": _f(self.nll_no_dnl_A),
            "nll_self_dnl_A": _f(self.nll_self_dnl_A),
            "nll_cross_dnl_A": _f(self.nll_cross_dnl_A),
            "nll_no_dnl_B": _f(self.nll_no_dnl_B),
            "nll_self_dnl_B": _f(self.nll_self_dnl_B),
            "nll_cross_dnl_B": _f(self.nll_cross_dnl_B),
            "cross_improvement_A": _f(self.cross_improvement_A),
            "cross_improvement_B": _f(self.cross_improvement_B),
            "self_improvement_A": _f(self.self_improvement_A),
            "self_improvement_B": _f(self.self_improvement_B),
            "overfitting_indicator_A": _f(self.overfitting_indicator_A),
            "overfitting_indicator_B": _f(self.overfitting_indicator_B),
            "dnl_correlation": _f(self.dnl_correlation),
            "verdict": self.verdict,
            "verdict_reasons": self.verdict_reasons,
        }
        return d


def _f(v: float) -> float | None:
    """Convert NaN to None for JSON serialisation."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _seed_priors_from_fit(priors: dict, fit_result) -> dict:
    """Update prior means to best-fit values from a previous fit.

    This seeds the Minuit initialisation at a known good minimum,
    preventing the DNL fits from landing in a different (worse) basin.
    The prior widths (sigmas) are preserved so the fit retains the
    same flexibility.
    """
    from fitting import FitResult
    if isinstance(fit_result, FitResult):
        params = fit_result.params
    elif isinstance(fit_result, dict):
        params = fit_result
    else:
        return dict(priors)
    seeded = dict(priors)
    for name, (mean, sigma) in priors.items():
        if name in params and not name.startswith("d"):
            val = params[name]
            if isinstance(val, (int, float)) and np.isfinite(val):
                seeded[name] = (float(val), sigma)
    return seeded


def _extract_nll(fit_result) -> float:
    """Extract NLL from a FitResult or dict."""
    from fitting import FitResult
    if isinstance(fit_result, FitResult):
        return float(fit_result.params.get("nll", np.nan))
    if isinstance(fit_result, dict):
        return float(fit_result.get("nll", np.nan))
    return np.nan


def _extract_dnl_factors(fit_result) -> np.ndarray | None:
    """Extract DNL factors from a FitResult or dict."""
    from fitting import FitResult
    if isinstance(fit_result, FitResult):
        params = fit_result.params
    elif isinstance(fit_result, dict):
        params = fit_result
    else:
        return None
    dnl_meta = params.get("_dnl", {})
    factors = dnl_meta.get("dnl_factors")
    if factors is not None:
        return np.asarray(factors, dtype=float)
    return None


def run_dnl_crossval(
    energies: np.ndarray,
    timestamps: np.ndarray | None,
    fit_kwargs: dict[str, Any],
    cfg: dict[str, Any],
) -> DNLCrossValResult:
    """Run DNL cross-validation on time-split halves.

    Parameters
    ----------
    energies : ndarray
        Energy values (MeV) for all events.
    timestamps : ndarray or None
        Timestamps for time-ordered splitting.  If None, uses index-based
        first-half / second-half split.
    fit_kwargs : dict
        Keyword arguments for ``fit_spectrum`` (priors, flags, bins,
        bin_edges, bounds, etc.).  Must NOT include ``energies``.
    cfg : dict
        Full pipeline configuration dict.

    Returns
    -------
    DNLCrossValResult
    """
    from fitting import FitResult, fit_spectrum

    result = DNLCrossValResult()
    n = len(energies)
    if n < 200:
        result.verdict = "insufficient_data"
        result.verdict_reasons.append(f"Only {n} events; need >= 200 for crossval")
        logger.warning("DNL crossval: insufficient data (%d events)", n)
        return result

    # Split into two time halves
    if timestamps is not None:
        order = np.argsort(timestamps)
    else:
        order = np.arange(n)
    mid = n // 2
    idx_A = order[:mid]
    idx_B = order[mid:]
    E_A = energies[idx_A]
    E_B = energies[idx_B]
    logger.info(
        "DNL crossval: splitting %d events into %d (A) + %d (B)",
        n, len(E_A), len(E_B),
    )

    # Build fit kwargs for DNL-disabled and DNL-enabled fits
    def _make_cfg_no_dnl(base_cfg: dict) -> dict:
        c = copy.deepcopy(base_cfg)
        sf = c.setdefault("spectral_fit", {})
        dc = sf.setdefault("dnl_correction", {})
        dc["enabled"] = False
        return c

    def _make_flags(base_flags: dict, cfg_override: dict,
                    external_dnl: np.ndarray | None = None) -> dict:
        f = dict(base_flags)
        f["cfg"] = cfg_override
        if external_dnl is not None:
            f["external_dnl_factors"] = external_dnl
        else:
            f.pop("external_dnl_factors", None)
        return f

    base_flags = dict(fit_kwargs.get("flags", {}))
    common_kw = {
        k: v for k, v in fit_kwargs.items()
        if k not in ("energies", "flags")
    }
    # Always skip MINOS in crossval fits (6 fits; only NLL matters, not errors)
    common_kw["skip_minos"] = True
    _cv_be = common_kw.get("bin_edges")
    _cv_bins = common_kw.get("bins")
    logger.info(
        "DNL crossval: bin_edges=%s, bins=%s, common_kw keys=%s",
        f"array({len(_cv_be)})" if _cv_be is not None else "None",
        _cv_bins,
        list(common_kw.keys()),
    )
    cfg_no_dnl = _make_cfg_no_dnl(cfg)
    cfg_dnl = copy.deepcopy(cfg)  # DNL enabled as-is

    # ── Fit half A ──────────────────────────────────────────────
    logger.info("DNL crossval: fitting half A without DNL")
    try:
        fit_A_no = fit_spectrum(
            E_A,
            flags=_make_flags(base_flags, cfg_no_dnl),
            **common_kw,
        )
        result.nll_no_dnl_A = _extract_nll(fit_A_no)
    except Exception as e:
        logger.warning("DNL crossval: half A no-DNL fit failed: %s", e)
        fit_A_no = None

    # Seed DNL fits from no-DNL results to avoid local-minimum traps
    common_kw_A = dict(common_kw)
    if fit_A_no is not None:
        common_kw_A["priors"] = _seed_priors_from_fit(
            common_kw.get("priors", {}), fit_A_no)

    logger.info("DNL crossval: fitting half A with self-DNL")
    try:
        fit_A_self = fit_spectrum(
            E_A,
            flags=_make_flags(base_flags, cfg_dnl),
            **common_kw_A,
        )
        result.nll_self_dnl_A = _extract_nll(fit_A_self)
        result.dnl_factors_A = _extract_dnl_factors(fit_A_self)
        logger.info(
            "DNL crossval: half A self-DNL NLL=%.1f, factors=%s",
            result.nll_self_dnl_A,
            "extracted (%d bins)" % len(result.dnl_factors_A)
            if result.dnl_factors_A is not None else "NONE",
        )
    except Exception as e:
        logger.warning("DNL crossval: half A self-DNL fit failed: %s\n%s",
                        e, traceback.format_exc())
        fit_A_self = None

    # ── Fit half B ──────────────────────────────────────────────
    logger.info("DNL crossval: fitting half B without DNL")
    try:
        fit_B_no = fit_spectrum(
            E_B,
            flags=_make_flags(base_flags, cfg_no_dnl),
            **common_kw,
        )
        result.nll_no_dnl_B = _extract_nll(fit_B_no)
    except Exception as e:
        logger.warning("DNL crossval: half B no-DNL fit failed: %s", e)
        fit_B_no = None

    common_kw_B = dict(common_kw)
    if fit_B_no is not None:
        common_kw_B["priors"] = _seed_priors_from_fit(
            common_kw.get("priors", {}), fit_B_no)

    logger.info("DNL crossval: fitting half B with self-DNL")
    try:
        fit_B_self = fit_spectrum(
            E_B,
            flags=_make_flags(base_flags, cfg_dnl),
            **common_kw_B,
        )
        result.nll_self_dnl_B = _extract_nll(fit_B_self)
        result.dnl_factors_B = _extract_dnl_factors(fit_B_self)
        logger.info(
            "DNL crossval: half B self-DNL NLL=%.1f, factors=%s",
            result.nll_self_dnl_B,
            "extracted (%d bins)" % len(result.dnl_factors_B)
            if result.dnl_factors_B is not None else "NONE",
        )
    except Exception as e:
        logger.warning("DNL crossval: half B self-DNL fit failed: %s\n%s",
                        e, traceback.format_exc())
        fit_B_self = None

    # ── Cross-apply DNL ─────────────────────────────────────────
    if result.dnl_factors_A is not None:
        logger.info(
            "DNL crossval: fitting half B with A's DNL factors "
            "(shape=%s, range=[%.4f, %.4f])",
            result.dnl_factors_A.shape,
            result.dnl_factors_A.min(),
            result.dnl_factors_A.max(),
        )
        try:
            fit_B_cross = fit_spectrum(
                E_B,
                flags=_make_flags(base_flags, cfg_dnl,
                                  external_dnl=result.dnl_factors_A),
                **common_kw_B,
            )
            result.nll_cross_dnl_B = _extract_nll(fit_B_cross)
            logger.info("DNL crossval: half B cross-DNL NLL=%.1f",
                        result.nll_cross_dnl_B)
        except Exception as e:
            logger.error(
                "DNL crossval: half B cross-DNL fit failed: %s\n%s",
                e, traceback.format_exc(),
            )

    if result.dnl_factors_B is not None:
        logger.info(
            "DNL crossval: fitting half A with B's DNL factors "
            "(shape=%s, range=[%.4f, %.4f])",
            result.dnl_factors_B.shape,
            result.dnl_factors_B.min(),
            result.dnl_factors_B.max(),
        )
        try:
            fit_A_cross = fit_spectrum(
                E_A,
                flags=_make_flags(base_flags, cfg_dnl,
                                  external_dnl=result.dnl_factors_B),
                **common_kw_A,
            )
            result.nll_cross_dnl_A = _extract_nll(fit_A_cross)
            logger.info("DNL crossval: half A cross-DNL NLL=%.1f",
                        result.nll_cross_dnl_A)
        except Exception as e:
            logger.error(
                "DNL crossval: half A cross-DNL fit failed: %s\n%s",
                e, traceback.format_exc(),
            )

    # ── Compute derived metrics ─────────────────────────────────
    result.cross_improvement_A = result.nll_no_dnl_A - result.nll_cross_dnl_A
    result.cross_improvement_B = result.nll_no_dnl_B - result.nll_cross_dnl_B
    result.self_improvement_A = result.nll_no_dnl_A - result.nll_self_dnl_A
    result.self_improvement_B = result.nll_no_dnl_B - result.nll_self_dnl_B
    result.overfitting_indicator_A = (
        result.self_improvement_A - result.cross_improvement_A
    )
    result.overfitting_indicator_B = (
        result.self_improvement_B - result.cross_improvement_B
    )

    # DNL factor correlation (over bins where both are non-trivial)
    if result.dnl_factors_A is not None and result.dnl_factors_B is not None:
        fa = result.dnl_factors_A
        fb = result.dnl_factors_B
        if fa.shape == fb.shape:
            valid = (fa != 1.0) | (fb != 1.0)
            if np.sum(valid) > 10:
                corr = np.corrcoef(fa[valid], fb[valid])
                result.dnl_correlation = float(corr[0, 1])

    # ── Verdict ─────────────────────────────────────────────────
    reasons = []
    cross_ok_A = not np.isnan(result.cross_improvement_A) and result.cross_improvement_A > 0
    cross_ok_B = not np.isnan(result.cross_improvement_B) and result.cross_improvement_B > 0
    high_corr = not np.isnan(result.dnl_correlation) and result.dnl_correlation > 0.5
    low_corr = not np.isnan(result.dnl_correlation) and result.dnl_correlation < 0.3

    if high_corr:
        reasons.append(
            f"DNL factors correlate well between halves (r={result.dnl_correlation:.3f})"
        )
    elif low_corr:
        reasons.append(
            f"DNL factors correlate poorly between halves (r={result.dnl_correlation:.3f})"
        )
    else:
        reasons.append(
            f"DNL factor correlation is moderate (r={result.dnl_correlation:.3f})"
        )

    if cross_ok_A and cross_ok_B:
        reasons.append(
            f"Cross-DNL improves held-out NLL for both halves "
            f"(A: {result.cross_improvement_A:.1f}, B: {result.cross_improvement_B:.1f})"
        )
    elif cross_ok_A or cross_ok_B:
        reasons.append("Cross-DNL improves held-out NLL for one half only")
    else:
        reasons.append("Cross-DNL does not improve held-out NLL for either half")

    if not np.isnan(result.overfitting_indicator_A):
        avg_overfit = 0.5 * (
            result.overfitting_indicator_A + result.overfitting_indicator_B
        )
        if avg_overfit > 0:
            reasons.append(
                f"Self-DNL exceeds cross-DNL by avg {avg_overfit:.1f} NLL units "
                f"(potential model-misfit absorption)"
            )

    # Classification
    if high_corr and cross_ok_A and cross_ok_B:
        result.verdict = "hardware_signal"
    elif low_corr or (not cross_ok_A and not cross_ok_B):
        result.verdict = "overfitting"
    else:
        result.verdict = "mixed"

    result.verdict_reasons = reasons
    logger.info("DNL crossval verdict: %s  - %s", result.verdict, "; ".join(reasons))
    return result
