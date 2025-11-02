"""Utilities for resolving EMG tail usage in spectral fits.

This module centralises the precedence rules for enabling exponentially
modified Gaussian (EMG) tails and for selecting the initial tail constant.
Call :func:`resolve_emg_tail_usage` with the priors and configuration flags
passed to the spectral fitter to obtain a per-isotope decision.  The
precedence order mirrors the documented behaviour:

1. Explicit ``tau`` priors always enable the corresponding tail.
2. A ``use_emg`` mapping may enable or disable individual isotopes.
3. A scalar ``use_emg`` flag toggles every isotope that was not already
   decided by the previous rules.
4. When none of the above apply the tail remains disabled.

Each decision provides a ``tau_hint`` that may be used as a starting value if
no prior was supplied.  The hint is clipped to the provided ``tau_floor`` to
avoid the numerical issues caused by extremely small values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import math

__all__ = ["EMGTailDecision", "resolve_emg_tail_usage"]


@dataclass(frozen=True)
class EMGTailDecision:
    """Resolved EMG tail preference for a single isotope."""

    enabled: bool
    tau_hint: float | None
    source: str | None = None


def _get_flag_value(flags: Mapping[str, Any] | SimpleNamespace | None, key: str) -> Any:
    if flags is None:
        return None
    if isinstance(flags, Mapping):
        return flags.get(key)
    return getattr(flags, key, None)


def _coerce_tau(value: Any, tau_floor: float) -> float:
    try:
        tau = float(value)
    except (TypeError, ValueError):
        return float(tau_floor)
    if not math.isfinite(tau):
        return float(tau_floor)
    return float(max(tau, tau_floor))


def resolve_emg_tail_usage(
    priors: Mapping[str, Any],
    *,
    flags: Mapping[str, Any] | SimpleNamespace | None = None,
    isotopes: Sequence[str] | None = None,
    tau_floor: float = 1e-8,
) -> dict[str, EMGTailDecision]:
    """Return per-isotope EMG decisions applying the documented precedence."""

    if isotopes is None:
        isotopes = ("Po210", "Po218", "Po214")

    tau_floor = float(tau_floor)
    decisions: dict[str, EMGTailDecision] = {}
    use_emg = {iso: False for iso in isotopes}
    tau_hints = {iso: tau_floor for iso in isotopes}
    sources = {iso: "default" for iso in isotopes}
    forced_true: set[str] = set()

    string_types = (str, bytes)

    for iso in isotopes:
        tau_key = f"tau_{iso}"
        prior = priors.get(tau_key)
        if prior is None:
            continue
        tau_value: float | None = None
        if isinstance(prior, Sequence) and not isinstance(prior, string_types) and prior:
            tau_value = _coerce_tau(prior[0], tau_floor)
        else:
            tau_value = _coerce_tau(prior, tau_floor)
        tau_hints[iso] = tau_value
        use_emg[iso] = True
        sources[iso] = "prior"
        forced_true.add(iso)

    use_flag = _get_flag_value(flags, "use_emg")
    if isinstance(use_flag, Mapping):
        for iso, value in use_flag.items():
            if iso not in use_emg or iso in forced_true:
                continue
            enabled = bool(value)
            use_emg[iso] = enabled
            sources[iso] = "config_map"
            if enabled:
                tau_hints[iso] = _coerce_tau(tau_hints[iso], tau_floor)
    elif use_flag is not None:
        enabled = bool(use_flag)
        for iso in isotopes:
            if iso in forced_true:
                continue
            use_emg[iso] = enabled
            sources[iso] = "config_scalar"
            if enabled:
                tau_hints[iso] = _coerce_tau(tau_hints[iso], tau_floor)

    for iso in isotopes:
        tau_hint = tau_hints[iso] if use_emg[iso] else None
        source = sources[iso]
        if not use_emg[iso] and source == "default":
            source = "default_off"
        decisions[iso] = EMGTailDecision(use_emg[iso], tau_hint, source)

    return decisions
