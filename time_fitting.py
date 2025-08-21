from __future__ import annotations

import logging
from typing import Any, Mapping

from fitting import fit_time_series, FitResult


def two_pass_time_fit(
    times_dict: Mapping[str, Any],
    t_start: float,
    t_end: float,
    fit_cfg: Mapping[str, Any],
    *,
    baseline_rate: float | None = None,
    background_b_fixed_value: float | None = None,
    fix_first_pass: bool = True,
    weights=None,
    strict: bool = False,
    fit_func=fit_time_series,
) -> FitResult:
    """Run a two-pass time fit with optional background fixing.

    Pass 1 fixes the background term ``B`` to ``background_b_fixed_value`` if
    provided, otherwise ``baseline_rate``. Pass 2 releases ``B`` and is kept
    only when it improves the AIC by at least 0.5. When ``fix_first_pass`` is
    ``False`` a single call to :func:`fit_time_series` is performed.
    """

    if not fix_first_pass:
        return fit_func(
            times_dict, t_start, t_end, fit_cfg, weights=weights, strict=strict
        )

    b_val = background_b_fixed_value
    if b_val is None:
        b_val = baseline_rate

    cfg1 = dict(fit_cfg)
    cfg1["fit_background"] = False
    cfg1["background_guess"] = float(b_val or 0.0)
    res1 = fit_func(
        times_dict, t_start, t_end, cfg1, weights=weights, strict=strict
    )

    cfg2 = dict(fit_cfg)
    cfg2["fit_background"] = True
    cfg2["background_guess"] = float(b_val or 0.0)
    res2 = fit_func(
        times_dict, t_start, t_end, cfg2, weights=weights, strict=strict
    )

    def _aic(res: FitResult) -> float:
        k = len(res.param_index or {})
        nll = res.params.get("nll", 0.0)
        return 2.0 * k + 2.0 * nll

    aic1 = _aic(res1)
    aic2 = _aic(res2)
    if aic2 <= aic1 - 0.5:
        return res2
    logging.debug(
        "two_pass_time_fit: keeping first-pass result (AIC1=%.3f, AIC2=%.3f)",
        aic1,
        aic2,
    )
    return res1
