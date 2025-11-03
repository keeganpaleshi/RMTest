"""Helpers for time-series fitting."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from math import isfinite

from fitting import fit_time_series, FitResult

TIME_FIT_CALLBACK_VERSION: tuple[int, int] = (1, 1)
"""Current version of the time-fit callback interface.

Callbacks that support keyword forwarding should expose their supported
interface by defining ``__rmtest_time_fit_callback_version__`` and setting it to
at least :data:`TIME_FIT_CALLBACK_VERSION`.
"""

_LEGACY_CALLBACK_VERSION: tuple[int, int] = (1, 0)
_CALLBACK_VERSION_ATTR = "__rmtest_time_fit_callback_version__"


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
    fit_kwargs: Mapping[str, object] | None = None,
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
        if fixed_val is not None and iso:
            eff_val = _prior_efficiency(config, iso)
            if eff_val is not None:
                fixed_val = float(fixed_val) * eff_val

    extra_kwargs = dict(fit_kwargs or {})

    def _resolve_attr(func, attr):
        seen: set[int] = set()
        current = func
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, attr):
                return getattr(current, attr)
            current = getattr(current, "__wrapped__", None) or getattr(current, "func", None)
        return None

    def _normalize_version(value) -> tuple[int, int]:
        if value is None:
            return _LEGACY_CALLBACK_VERSION
        if isinstance(value, (tuple, list)) and value:
            try:
                major = int(value[0])
                minor = int(value[1]) if len(value) > 1 else 0
                return (major, minor)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return _LEGACY_CALLBACK_VERSION
        if isinstance(value, str):  # pragma: no cover - defensive
            parts = value.replace("_", ".").split(".")
            try:
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                return (major, minor)
            except (ValueError, IndexError):
                return _LEGACY_CALLBACK_VERSION
        if isinstance(value, int):  # pragma: no cover - defensive
            return (int(value), 0)
        return _LEGACY_CALLBACK_VERSION

    callback_version = _normalize_version(_resolve_attr(fit_func, _CALLBACK_VERSION_ATTR))
    supports_extra_kwargs = callback_version >= TIME_FIT_CALLBACK_VERSION

    def _callable_keywords(func) -> tuple[set[str], bool]:
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):  # pragma: no cover - signature unavailable
            return set(), False

        accepts_var_kw = False
        allowed: set[str] = set()
        for param in sig.parameters.values():
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                accepts_var_kw = True
                continue
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                allowed.add(param.name)
        return allowed, accepts_var_kw

    if extra_kwargs:
        allowed_names, has_var_kw = _callable_keywords(fit_func)
        if not has_var_kw:
            forwarded = {k: v for k, v in extra_kwargs.items() if k in allowed_names}
            dropped = sorted(set(extra_kwargs) - set(forwarded))
        else:
            forwarded = extra_kwargs
            dropped = []

        if supports_extra_kwargs:
            extra_kwargs = forwarded
            if dropped:
                logging.debug(
                    "two_pass_time_fit: callback %r cannot handle fit_kwargs %s; dropping",
                    getattr(fit_func, "__name__", repr(fit_func)),
                    dropped,
                )
        else:
            if extra_kwargs:
                logging.debug(
                    "two_pass_time_fit: legacy callback %r; suppressing forwarded fit_kwargs",
                    getattr(fit_func, "__name__", repr(fit_func)),
                )
            extra_kwargs = {}

    if not fix_first or fixed_val is None or not iso:
        return fit_func(
            times_dict,
            t_start,
            t_end,
            config,
            weights=weights,
            strict=strict,
            **extra_kwargs,
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
        **extra_kwargs,
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
        **extra_kwargs,
    )

    def _aic(res: FitResult) -> float:
        k = len(res.param_index or {})
        nll = res.params.get("nll", 0.0)
        return 2.0 * k + 2.0 * nll

    valid1 = bool(res1.params.get("fit_valid", True))
    valid2 = bool(res2.params.get("fit_valid", True))

    if valid2 and (not valid1 or _aic(res1) - _aic(res2) >= 0.5):
        return res2
    return res1
