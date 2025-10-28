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

    fix_first = bool(config.get("fix_background_b_first_pass", True))
    fixed_val = config.get("background_b_fixed_value")
    baseline_fixed = False
    if fixed_val is None:
        fixed_val = baseline_rate
        baseline_fixed = baseline_rate is not None
        if fixed_val is not None and iso:
            eff_val = _prior_efficiency(config, iso)
            if eff_val is not None:
                fixed_val = float(fixed_val) * eff_val

    def _attach_background_metadata(result: FitResult, fixed_from_baseline: bool) -> FitResult:
        metadata = {
            "background_fixed_from_baseline": bool(fixed_from_baseline),
            "background_baseline_rate_Bq": None,
        }
        if fixed_from_baseline and baseline_rate is not None:
            try:
                metadata["background_baseline_rate_Bq"] = float(baseline_rate)
            except (TypeError, ValueError):
                metadata["background_baseline_rate_Bq"] = None

        result.background_metadata = metadata  # type: ignore[attr-defined]

        if isinstance(result.params, dict):
            result.params["background_fixed_from_baseline"] = metadata[
                "background_fixed_from_baseline"
            ]
            if metadata["background_fixed_from_baseline"] and metadata[
                "background_baseline_rate_Bq"
            ] is not None:
                result.params["background_baseline_rate_Bq"] = metadata[
                    "background_baseline_rate_Bq"
                ]
            else:
                result.params.pop("background_baseline_rate_Bq", None)

        return result

    if not fix_first or fixed_val is None or not iso:
        return _attach_background_metadata(
            fit_func(
                times_dict,
                t_start,
                t_end,
                config,
                weights=weights,
                strict=strict,
            ),
            False,
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
        return _attach_background_metadata(res2, False)
    return _attach_background_metadata(res1, baseline_fixed)
