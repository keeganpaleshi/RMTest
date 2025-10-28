"""Helpers for time-series fitting."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from math import isfinite

from fitting import fit_time_series, FitResult


def _prior_efficiency(config: Mapping[str, object], iso: str) -> float | None:
    """Return the configured efficiency for ``iso`` if it is finite."""

    try:
        iso_cfg = config.get("isotopes", {}).get(iso, {})
    except AttributeError:
        return None

    eff_cfg = None
    if isinstance(iso_cfg, Mapping):
        eff_cfg = iso_cfg.get("efficiency")
    if isinstance(eff_cfg, (list, tuple)):
        eff_cfg = eff_cfg[0] if eff_cfg else None

    if eff_cfg is None:
        return None

    try:
        eff_val = float(eff_cfg)
    except (TypeError, ValueError):
        return None

    if not isfinite(eff_val) or eff_val <= 0:
        return None

    return eff_val


def two_pass_time_fit(
    times_dict: Mapping[str, list | tuple | object],
    t_start: float,
    t_end: float,
    config: dict,
    *,
    baseline_rate: float | None = None,
    weights=None,
    strict: bool = False,
    fit_func=fit_time_series,
) -> FitResult:
    """Run a two-pass time-series fit with optional fixed background.

    Pass 1 optionally fixes the background parameter ``B`` to a provided value
    before refitting with ``B`` free.  The second pass is kept only if its
    Akaike Information Criterion (AIC) improves by at least 0.5.
    """

    iso_list = list(times_dict.keys())
    iso = iso_list[0] if iso_list else ""

    baseline_rate_value = float(baseline_rate) if baseline_rate is not None else None

    def _annotate(res_obj, strategy: str, baseline_val: float | None = None):
        """Attach background-handling metadata to ``res_obj``."""

        baseline_numeric = (
            float(baseline_val) if baseline_val is not None else None
        )
        params = getattr(res_obj, "params", None)

        if isinstance(params, Mapping):
            try:
                params["background_strategy"] = strategy
                if baseline_numeric is not None:
                    params["baseline_rate_used_Bq"] = baseline_numeric
                else:
                    params.pop("baseline_rate_used_Bq", None)
            except TypeError:
                params = dict(params)
                params["background_strategy"] = strategy
                if baseline_numeric is not None:
                    params["baseline_rate_used_Bq"] = baseline_numeric
                else:
                    params.pop("baseline_rate_used_Bq", None)
                res_obj.params = params  # type: ignore[attr-defined]
        elif isinstance(res_obj, dict):
            res_obj["background_strategy"] = strategy
            if baseline_numeric is not None:
                res_obj["baseline_rate_used_Bq"] = baseline_numeric
            else:
                res_obj.pop("baseline_rate_used_Bq", None)

        return res_obj

    fix_first = bool(config.get("fix_background_b_first_pass", True))
    fixed_val = config.get("background_b_fixed_value")
    if fixed_val is None:
        fixed_val = baseline_rate
        if fixed_val is not None and iso:
            eff_val = _prior_efficiency(config, iso)
            if eff_val is not None:
                fixed_val = float(fixed_val) * eff_val

    if not fix_first or fixed_val is None or not iso:
        result = fit_func(
            times_dict,
            t_start,
            t_end,
            config,
            weights=weights,
            strict=strict,
        )
        if bool(config.get("fit_background", True)):
            return _annotate(result, "floated")
        strategy = (
            "fixed_from_baseline"
            if baseline_rate_value is not None
            else "fixed"
        )
        return _annotate(
            result,
            strategy,
            baseline_rate_value if strategy == "fixed_from_baseline" else None,
        )

    cfg_first = dict(config)
    cfg_first["fit_background"] = False
    res1 = fit_func(
        times_dict,
        t_start,
        t_end,
        cfg_first,
        weights=weights,
        strict=strict,
        fixed_background={iso: float(fixed_val)},
    )

    cfg_second = dict(config)
    cfg_second["fit_background"] = True
    res2 = fit_func(
        times_dict,
        t_start,
        t_end,
        cfg_second,
        weights=weights,
        strict=strict,
    )

    def _aic(res: FitResult) -> float:
        k = len(res.param_index or {})
        nll = res.params.get("nll", 0.0)
        return 2.0 * k + 2.0 * nll

    valid1 = bool(res1.params.get("fit_valid", True))
    valid2 = bool(res2.params.get("fit_valid", True))

    if valid2 and (not valid1 or _aic(res1) - _aic(res2) >= 0.5):
        return _annotate(res2, "floated")

    strategy_first = (
        "fixed_from_baseline" if baseline_rate_value is not None else "fixed"
    )
    return _annotate(
        res1,
        strategy_first,
        baseline_rate_value if strategy_first == "fixed_from_baseline" else None,
    )
