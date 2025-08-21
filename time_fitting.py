"""Two-pass time fit utilities."""
from __future__ import annotations

from typing import Mapping, Dict, Any
import logging

from fitting import fit_time_series as _default_fit_time_series, _neg_log_likelihood_time, FitResult

logger = logging.getLogger(__name__)

__all__ = ["two_pass_time_fit"]


def _compute_aic(
    fit: FitResult,
    times_dict: Mapping[str, Any],
    t_start: float,
    t_end: float,
    iso_list,
    lam_map,
    eff_map,
    fix_b_map,
    fix_n0_map,
    weights_dict,
) -> float:
    """Compute AIC for a given ``FitResult``."""
    if not fit.param_index:
        return float("inf")
    ordered = [n for n, _ in sorted(fit.param_index.items(), key=lambda kv: kv[1])]
    params = [fit.params.get(name, 0.0) for name in ordered]
    nll = _neg_log_likelihood_time(
        params,
        times_dict,
        weights_dict,
        t_start,
        t_end,
        iso_list,
        lam_map,
        eff_map,
        fix_b_map,
        fix_n0_map,
        fit.param_index,
    )
    k = len(fit.param_index)
    return 2.0 * k + 2.0 * nll


def two_pass_time_fit(
    times_dict: Mapping[str, Any],
    t_start: float,
    t_end: float,
    fit_cfg: Dict[str, Any],
    *,
    weights: Mapping[str, Any] | None = None,
    strict: bool = False,
    baseline_rate: float | None = None,
    fit_func=_default_fit_time_series,
) -> FitResult:
    """Run a two-pass time fit with optional fixed background.

    Parameters
    ----------
    times_dict : mapping
        Isotope -> array of event times in seconds.
    t_start, t_end : float
        Absolute start and end times in seconds.
    fit_cfg : dict
        Configuration for :func:`fitting.fit_time_series`. Expected keys are
        the same as that function plus the custom ``fix_background_b_first_pass``
        and ``background_b_fixed_value`` options.
    weights : mapping, optional
        Optional weights matching ``times_dict``.
    strict : bool, optional
        When ``True`` propagate covariance failures.
    baseline_rate : float, optional
        Baseline background rate to use when ``background_b_fixed_value`` is
        unspecified.
    """
    tf_opts = {
        "isotopes": fit_cfg.get("isotopes", {}),
        "fit_background": fit_cfg.get("fit_background", False),
        "fit_initial": fit_cfg.get("fit_initial", False),
        "background_guess": fit_cfg.get("background_guess", 0.0),
        "n0_guess_fraction": fit_cfg.get("n0_guess_fraction", 0.1),
        "min_counts": fit_cfg.get("min_counts"),
    }

    fix_first = bool(fit_cfg.get("fix_background_b_first_pass", True))
    b_fixed = fit_cfg.get("background_b_fixed_value")
    if b_fixed is None:
        b_fixed = baseline_rate

    # Pass 1: background fixed if requested
    if fix_first and b_fixed is not None:
        pass1_cfg = dict(tf_opts)
        pass1_cfg["fit_background"] = False
        pass1_cfg["background_guess"] = float(b_fixed)
    else:
        pass1_cfg = dict(tf_opts)

    weights_dict = weights
    decay1 = fit_func(
        times_dict,
        t_start,
        t_end,
        pass1_cfg,
        weights=weights_dict,
        strict=strict,
    )

    if not fix_first:
        return decay1

    pass2_cfg = dict(tf_opts)
    pass2_cfg["fit_background"] = True
    if b_fixed is not None:
        pass2_cfg["background_guess"] = float(b_fixed)

    decay2 = fit_func(
        times_dict,
        t_start,
        t_end,
        pass2_cfg,
        weights=weights_dict,
        strict=strict,
    )

    # Prepare shared maps for AIC calculation
    iso_list = list(tf_opts["isotopes"].keys())
    lam_map = {}
    eff_map = {}
    fix_b_map1 = {}
    fix_n0_map1 = {}
    fix_b_map2 = {}
    fix_n0_map2 = {}
    for iso, cfg_iso in tf_opts["isotopes"].items():
        hl = float(cfg_iso["half_life_s"])
        lam_map[iso] = __import__("math").log(2.0) / hl
        eff = cfg_iso.get("efficiency")
        eff_map[iso] = None if eff is None else float(eff)
        fix_b_map1[iso] = not bool(pass1_cfg.get("fit_background", False))
        fix_n0_map1[iso] = not bool(pass1_cfg.get("fit_initial", False))
        fix_b_map2[iso] = not bool(pass2_cfg.get("fit_background", False))
        fix_n0_map2[iso] = not bool(pass2_cfg.get("fit_initial", False))

    weights_dict = {
        iso: None if weights is None else weights.get(iso)
        for iso in iso_list
    }

    aic1 = _compute_aic(
        decay1,
        times_dict,
        t_start,
        t_end,
        iso_list,
        lam_map,
        eff_map,
        fix_b_map1,
        fix_n0_map1,
        weights_dict,
    )
    aic2 = _compute_aic(
        decay2,
        times_dict,
        t_start,
        t_end,
        iso_list,
        lam_map,
        eff_map,
        fix_b_map2,
        fix_n0_map2,
        weights_dict,
    )

    if aic2 <= aic1 - 0.5:
        logger.debug("two_pass_time_fit: using pass2 (AIC %.3f < %.3f)", aic2, aic1)
        return decay2

    logger.debug("two_pass_time_fit: keeping pass1 (AIC %.3f <= %.3f)", aic2, aic1)
    return decay1

