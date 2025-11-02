"""Helpers for resolving the stable EMG mode preference.

This module provides a small state container used to keep the various
`rmtest` modules in sync when deciding whether the numerically stable EMG
implementation should be preferred.  Consumers may feed a configuration
mapping (loaded from YAML or a namespace object) via
``set_emg_mode_from_config`` and query the current resolved value with
``get_emg_stable_mode``.  The precedence order is:

1. an explicit runtime override set via :func:`set_emg_mode_override`;
2. the configuration supplied to :func:`set_emg_mode_from_config`;
3. the global default (``True``).

The helper intentionally accepts both mappings and ``SimpleNamespace``
objects as this mirrors the structures returned throughout the codebase.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from ..emg_constants import EMG_STABLE_MODE

__all__ = [
    "get_emg_stable_mode",
    "resolve_emg_mode_preference",
    "set_emg_mode_from_config",
    "set_emg_mode_override",
    "reset_emg_mode_preferences",
]

_DEFAULT_PREFERENCE = EMG_STABLE_MODE
_config_preference: bool | None = None
_runtime_override: bool | None = None


def _coerce_bool(value: object | None) -> bool | None:
    """Return ``value`` coerced to ``bool`` when possible.

    ``None`` is propagated to signal that no preference was supplied.  The
    coercion purposefully mimics the legacy behaviour which treated any
    truthy value as ``True`` and everything else as ``False`` while still
    allowing ``None`` to mean "use the default".
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _extract_fitting_cfg(cfg: Mapping[str, Any] | SimpleNamespace | None) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, Mapping):
        return cfg.get("fitting")
    return getattr(cfg, "fitting", None)


def _extract_config_preference(
    cfg: Mapping[str, Any] | SimpleNamespace | None,
) -> bool | None:
    fitting_cfg = _extract_fitting_cfg(cfg)
    if isinstance(fitting_cfg, Mapping):
        if "emg_stable_mode" in fitting_cfg:
            return _coerce_bool(fitting_cfg.get("emg_stable_mode"))
        if "use_stable_emg" in fitting_cfg:
            return _coerce_bool(fitting_cfg.get("use_stable_emg"))
    elif isinstance(fitting_cfg, SimpleNamespace):
        if hasattr(fitting_cfg, "emg_stable_mode"):
            return _coerce_bool(getattr(fitting_cfg, "emg_stable_mode"))
        if hasattr(fitting_cfg, "use_stable_emg"):
            return _coerce_bool(getattr(fitting_cfg, "use_stable_emg"))
    return None


def resolve_emg_mode_preference(
    *,
    override: bool | None = None,
    config: bool | None = None,
    default: bool = _DEFAULT_PREFERENCE,
) -> bool:
    """Resolve the EMG preference using a consistent precedence order."""

    if override is not None:
        return bool(override)
    if config is not None:
        return bool(config)
    return bool(default)


def set_emg_mode_from_config(
    cfg: Mapping[str, Any] | SimpleNamespace | None,
) -> bool:
    """Update the stored configuration preference and return the result."""

    global _config_preference
    _config_preference = _extract_config_preference(cfg)
    return get_emg_stable_mode()


def set_emg_mode_override(value: object | None) -> bool:
    """Apply a runtime override and return the resolved preference."""

    global _runtime_override
    _runtime_override = _coerce_bool(value)
    return get_emg_stable_mode()


def get_emg_stable_mode(default: bool = _DEFAULT_PREFERENCE) -> bool:
    """Return the resolved EMG stable mode preference."""

    return resolve_emg_mode_preference(
        override=_runtime_override,
        config=_config_preference,
        default=default,
    )


def reset_emg_mode_preferences() -> None:
    """Clear stored configuration and runtime overrides."""

    global _config_preference, _runtime_override
    _config_preference = None
    _runtime_override = None
