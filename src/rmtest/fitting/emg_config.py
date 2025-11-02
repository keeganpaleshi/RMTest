"""Helpers for resolving EMG configuration preferences.

This module centralises how the compatibility layer determines whether the
stable exponentially modified Gaussian (EMG) implementation should be used.
Configurations coming either from dictionaries or ``SimpleNamespace`` objects
share a single precedence function so that the core :mod:`fitting` module,
the :mod:`rmtest` compatibility package and any external callers all reach the
same decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

DEFAULT_EMG_STABLE_MODE: bool = True


@dataclass(frozen=True)
class EmgPreferences:
    """Resolved EMG configuration flags."""

    use_stable_emg: bool
    emg_stable_mode: bool


def _get_section(cfg: Mapping[str, Any] | SimpleNamespace | None, name: str) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, Mapping):
        return cfg.get(name)
    return getattr(cfg, name, None)


def _get_value(section: Any, key: str) -> Any:
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def resolve_emg_preferences(
    cfg: Mapping[str, Any] | SimpleNamespace | None,
    *,
    default_mode: bool = DEFAULT_EMG_STABLE_MODE,
) -> EmgPreferences:
    """Return the canonical EMG configuration derived from ``cfg``.

    ``default_mode`` mirrors the historic default of enabling the stable EMG
    backend.  When configuration data is absent the returned preferences fall
    back to this default.

    The precedence is:

    1. ``fitting.emg_stable_mode`` – when provided this value wins and forces
       ``use_stable_emg`` to match.
    2. ``fitting.use_stable_emg`` – honoured when the dedicated stable-mode
       toggle is absent.
    3. ``default_mode`` – used when neither key is defined.
    """

    section = _get_section(cfg, "fitting")

    use_value = default_mode
    mode_value = default_mode

    use_raw = _get_value(section, "use_stable_emg")
    coerced_use = _coerce_bool(use_raw)
    if coerced_use is not None:
        use_value = coerced_use
        mode_value = coerced_use

    mode_raw = _get_value(section, "emg_stable_mode")
    coerced_mode = _coerce_bool(mode_raw)
    if coerced_mode is not None:
        mode_value = coerced_mode
        use_value = coerced_mode

    return EmgPreferences(use_value, mode_value)


def apply_emg_preferences(
    cfg: Mapping[str, Any] | SimpleNamespace | None,
    *,
    default_mode: bool = DEFAULT_EMG_STABLE_MODE,
) -> EmgPreferences:
    """Resolve EMG flags from ``cfg`` and synchronise the core module state."""

    preferences = resolve_emg_preferences(cfg, default_mode=default_mode)

    try:
        import fitting as _core_fitting  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency path
        return preferences

    _core_fitting.EMG_STABLE_MODE = preferences.emg_stable_mode
    return preferences


__all__ = [
    "DEFAULT_EMG_STABLE_MODE",
    "EmgPreferences",
    "apply_emg_preferences",
    "resolve_emg_preferences",
]

