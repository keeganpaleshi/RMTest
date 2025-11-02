"""Utilities for resolving EMG usage within the fitting pipeline.

This module centralises the logic required to decide whether an EMG tail
should be enabled for a given isotope and which starting ``tau`` value should
be used when one is required.  The decision is based on three sources of
information, applied in a well-defined precedence order:

1. Explicit ``use_emg`` overrides supplied via the fit ``flags``.
2. The presence of a ``tau`` prior in the fit configuration.
3. A caller-provided default (falling back to the global minimum).

The helper is intentionally liberal with the accepted mapping types so that it
can be used with raw dictionaries, ``TypedDict`` instances, or
``SimpleNamespace`` objects, mirroring the structures used throughout the
codebase.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Iterator

DEFAULT_ISOTOPES: tuple[str, ...] = ("Po210", "Po218", "Po214")

__all__ = ["DEFAULT_ISOTOPES", "EmgTailSetting", "resolve_emg_tail_settings"]


@dataclass(frozen=True)
class EmgTailSetting:
    """Resolved EMG configuration for a single isotope."""

    enabled: bool
    tau: float | None = None


def _iter_isotopes(isotopes: Iterable[str] | None) -> Iterator[str]:
    if isotopes is None:
        yield from DEFAULT_ISOTOPES
    else:
        for iso in isotopes:
            yield str(iso)


def _coerce_tau(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None


def resolve_emg_tail_settings(
    *,
    priors: Mapping[str, Any] | None = None,
    flags: Mapping[str, Any] | None = None,
    isotopes: Iterable[str] | None = None,
    tau_floor: float = 0.0,
    default_tau: float | None = None,
) -> dict[str, EmgTailSetting]:
    """Return resolved EMG settings for the requested isotopes.

    Parameters
    ----------
    priors:
        Mapping containing prior tuples ``(mean, sigma)``.  Any prior for a
        ``tau`` parameter (``"tau_{iso}"``) automatically enables the EMG tail
        for that isotope.
    flags:
        Mapping that may contain a ``"use_emg"`` entry.  When present this
        overrides the automatic behaviour based on priors.  The entry may be a
        scalar boolean or a mapping of isotope names to booleans.
    isotopes:
        Iterable of isotope names to consider.  When omitted, the default
        ``("Po210", "Po218", "Po214")`` ordering is used.
    tau_floor:
        Lower bound applied to any resolved ``tau`` value.
    default_tau:
        Starting ``tau`` value used for isotopes without an explicit prior.

    Returns
    -------
    dict
        Mapping of isotope name to :class:`EmgTailSetting` objects.
    """

    priors = priors or {}
    use_emg_flag: Any | None = None
    if isinstance(flags, Mapping):
        use_emg_flag = flags.get("use_emg")

    tau_floor = float(tau_floor)
    default_tau = _coerce_tau(default_tau)

    forced_true: set[str] = set()
    tau_guesses: dict[str, float | None] = {}
    isotopes_tuple = tuple(_iter_isotopes(isotopes))
    use_emg_map: dict[str, bool] = {iso: False for iso in isotopes_tuple}

    for iso in isotopes_tuple:
        prior = priors.get(f"tau_{iso}")
        mean: float | None = None
        if isinstance(prior, Mapping):
            mean = _coerce_tau(prior.get("mean"))
        elif isinstance(prior, (tuple, list)):
            if prior:
                mean = _coerce_tau(prior[0])
        else:
            mean = _coerce_tau(prior)

        if mean is not None:
            use_emg_map[iso] = True
            forced_true.add(iso)
            tau_guesses[iso] = max(mean, tau_floor)

    if isinstance(use_emg_flag, Mapping):
        for iso, value in use_emg_flag.items():
            if iso not in use_emg_map:
                continue
            if iso in forced_true:
                continue
            use_emg_map[iso] = bool(value)
    elif use_emg_flag is not None:
        scalar = bool(use_emg_flag)
        for iso in isotopes_tuple:
            if iso in forced_true:
                continue
            use_emg_map[iso] = scalar

    resolved: dict[str, EmgTailSetting] = {}
    for iso in isotopes_tuple:
        enabled = use_emg_map.get(iso, False)
        tau_value: float | None = None
        if enabled:
            tau_value = tau_guesses.get(iso)
            if tau_value is None:
                tau_value = default_tau
            if tau_value is None:
                tau_value = tau_floor
            tau_value = max(float(tau_value), tau_floor)
        resolved[iso] = EmgTailSetting(enabled=enabled, tau=tau_value)

    return resolved

