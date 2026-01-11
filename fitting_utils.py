"""Utility functions for working with fit results and parameters.

This module provides helper functions for extracting parameters, efficiencies,
and uncertainties from fit results and configuration dictionaries.
"""

import math
from typing import Any, Mapping, cast

from fitting import FitResult, FitParams


def fit_params(obj: FitResult | Mapping[str, float] | None) -> FitParams:
    """Return fit parameters mapping from a ``FitResult`` or dictionary."""
    if isinstance(obj, FitResult):
        return cast(FitParams, obj.params)
    if isinstance(obj, Mapping):
        return obj  # type: ignore[return-value]
    return {}


def config_efficiency(cfg: Mapping[str, Any], iso: str) -> float:
    """Return the prior efficiency for ``iso`` from ``cfg``."""

    eff_cfg = cfg.get("time_fit", {}).get(f"eff_{iso.lower()}")
    if isinstance(eff_cfg, (list, tuple)):
        return float(eff_cfg[0]) if eff_cfg else 1.0
    if eff_cfg is None or eff_cfg == "null":
        return 1.0
    try:
        return float(eff_cfg)
    except (TypeError, ValueError):
        return 1.0


def fit_efficiency(params: Mapping[str, Any] | None, iso: str) -> float | None:
    """Return fitted efficiency for ``iso`` if present in ``params``."""

    if not params:
        return None

    keys = ("eff", f"eff_{iso}", f"eff_{iso.lower()}")
    for key in keys:
        val = params.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def resolved_efficiency(
    cfg: Mapping[str, Any], iso: str, params: Mapping[str, Any] | None
) -> float:
    """Return efficiency for ``iso`` preferring fitted values over priors."""

    fitted = fit_efficiency(params, iso)
    if fitted is not None and fitted > 0:
        return fitted
    return config_efficiency(cfg, iso)


def safe_float(value: Any) -> float | None:
    """Return ``value`` coerced to ``float`` when it is finite."""

    try:
        if value is None:
            return None
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def float_with_default(value: Any, default: float) -> float:
    """Return ``value`` as ``float`` or ``default`` when coercion fails."""

    coerced = safe_float(value)
    return default if coerced is None else coerced


def cov_lookup(
    fit_result: FitResult | Mapping[str, float] | None, name1: str, name2: str
) -> float:
    """Return covariance between two parameters if present."""
    if isinstance(fit_result, FitResult):
        try:
            return float(fit_result.cov_df.loc[name1, name2])
        except KeyError:
            try:
                return float(fit_result.get_cov(name1, name2))
            except KeyError:
                return 0.0
    if isinstance(fit_result, Mapping):
        return float(fit_result.get(f"cov_{name1}_{name2}", 0.0))
    return 0.0


def fallback_uncertainty(
    rate: float | None, fit_result: FitResult | Mapping[str, float] | None, param: str
) -> float:
    """Return uncertainty from covariance or a Poisson estimate."""

    def _try_var(value: Any) -> float | None:
        try:
            var_val = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(var_val) or var_val <= 0:
            return None
        return var_val

    candidates: list[Any] = []
    if isinstance(fit_result, FitResult):
        if fit_result.cov is not None and fit_result.param_index is not None:
            idx = fit_result.param_index.get(param)
            if idx is not None and idx < fit_result.cov.shape[0]:
                candidates.append(fit_result.cov[idx, idx])
        candidates.append(fit_result.params.get(f"cov_{param}_{param}"))
    elif isinstance(fit_result, Mapping):
        candidates.append(fit_result.get(f"cov_{param}_{param}"))

    for cand in candidates:
        var = _try_var(cand)
        if var is not None:
            return math.sqrt(var)

    try:
        rate_val = float(rate) if rate is not None else None
    except (TypeError, ValueError):
        rate_val = None

    if rate_val is None or not math.isfinite(rate_val):
        return 0.0

    return math.sqrt(abs(rate_val))
