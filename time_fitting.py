"""Helpers for time-series fitting."""

from __future__ import annotations

import logging
from typing import Mapping

from fitting import fit_time_series, FitResult


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
    if fixed_val is None:
        fixed_val = baseline_rate

    if not fix_first or fixed_val is None or not iso:
        return fit_func(
            times_dict,
            t_start,
            t_end,
            config,
            weights=weights,
            strict=strict,
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
        fixed_background={iso: fixed_val},
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

    if _aic(res1) - _aic(res2) >= 0.5:
        return res2
    return res1
