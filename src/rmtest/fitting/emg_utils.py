"""Utilities for resolving EMG configuration details used by the fitter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable

from .emg_config import (
    get_emg_stable_mode,
    resolve_emg_mode_preference,
    set_emg_mode_from_config,
    set_emg_mode_override,
)

__all__ = [
    "DEFAULT_ISOTOPES",
    "EMGTailSpec",
    "extract_use_emg_flag",
    "resolve_emg_mode_preference",
    "set_emg_mode_from_config",
    "set_emg_mode_override",
    "get_emg_stable_mode",
    "resolve_emg_tails",
]

# Default ordering matches the historical fitter expectations
DEFAULT_ISOTOPES: tuple[str, ...] = ("Po210", "Po218", "Po214")


@dataclass(frozen=True)
class EMGTailSpec:
    """Resolved EMG tail configuration for a single isotope."""

    enabled: bool
    tau: float


def extract_use_emg_flag(flags: Mapping[str, Any] | SimpleNamespace | None) -> Any:
    """Return the ``use_emg`` selector from *flags* if present."""

    if flags is None:
        return None
    if isinstance(flags, Mapping):
        return flags.get("use_emg")
    return getattr(flags, "use_emg", None)


def _coerce_mapping(value: Any, *, isotopes: Iterable[str]) -> dict[str, bool]:
    mapping: dict[str, bool] = {iso: False for iso in isotopes}
    if isinstance(value, Mapping):
        for iso, flag in value.items():
            if iso in mapping:
                mapping[iso] = bool(flag)
    else:
        scalar: bool | None
        if isinstance(value, bool):
            scalar = value
        elif value is None:
            scalar = None
        else:
            scalar = bool(value)
        if scalar is not None:
            mapping = {iso: scalar for iso in mapping}
    return mapping


def _extract_tau_prior_mean(value: Any) -> float | None:
    if isinstance(value, Mapping):
        if "mean" in value:
            try:
                return float(value["mean"])
            except (TypeError, ValueError):
                return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return None
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_emg_tails(
    priors: Mapping[str, Any] | None,
    flags: Mapping[str, Any] | SimpleNamespace | None = None,
    *,
    isotopes: Iterable[str] = DEFAULT_ISOTOPES,
    tau_floor: float,
    default_tau: float = 1.0,
) -> dict[str, EMGTailSpec]:
    """Resolve EMG enable flags and initial tau guesses for each isotope."""

    iso_list = tuple(isotopes)
    tau_defaults = {iso: max(float(default_tau), float(tau_floor)) for iso in iso_list}
    resolved: dict[str, EMGTailSpec] = {}
    priors = priors or {}

    forced_true: set[str] = set()
    for iso in iso_list:
        key = f"tau_{iso}"
        if key in priors:
            mean = _extract_tau_prior_mean(priors[key])
            if mean is not None:
                tau_defaults[iso] = max(float(mean), float(tau_floor))
            forced_true.add(iso)

    use_emg_cfg = extract_use_emg_flag(flags)
    use_emg_map = _coerce_mapping(use_emg_cfg, isotopes=iso_list)

    for iso in iso_list:
        enabled = iso in forced_true or use_emg_map.get(iso, False)
        tau_val = tau_defaults[iso] if enabled else 0.0
        resolved[iso] = EMGTailSpec(enabled=enabled, tau=tau_val)

    return resolved

