"""Helpers for managing the EMG stable mode configuration."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

DEFAULT_EMG_STABLE_MODE = True

_CONFIG_CACHE: Mapping[str, Any] | SimpleNamespace | None = None
_OVERRIDE: bool | None = None
_FALLBACK = DEFAULT_EMG_STABLE_MODE
_SENTINEL: object = object()

__all__ = [
    "DEFAULT_EMG_STABLE_MODE",
    "emg_stable_mode_from_config",
    "resolve_emg_stable_mode",
    "synchronize_emg_stable_mode",
]


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid EMG stable mode flag: {value!r}")
    if isinstance(value, (int, float)):
        if value in {0, 0.0}:
            return False
        if value in {1, 1.0}:
            return True
        raise ValueError(f"Invalid numeric EMG stable mode flag: {value!r}")
    raise TypeError(f"Unsupported EMG stable mode type: {type(value)!r}")


def _extract_fitting_section(
    config: Mapping[str, Any] | SimpleNamespace | None,
) -> Mapping[str, Any] | SimpleNamespace | None:
    if config is None:
        return None
    if isinstance(config, Mapping):
        section = config.get("fitting")
        if isinstance(section, Mapping) or isinstance(section, SimpleNamespace) or section is None:
            return section
        raise TypeError("'fitting' section must be a mapping or namespace")
    return getattr(config, "fitting", None)


def emg_stable_mode_from_config(
    config: Mapping[str, Any] | SimpleNamespace | None,
) -> bool | None:
    """Return the EMG stable mode flag extracted from ``config`` if present."""

    section = _extract_fitting_section(config)
    if section is None:
        return None
    raw: Any
    if isinstance(section, Mapping):
        if "emg_stable_mode" in section:
            raw = section.get("emg_stable_mode")
        else:
            raw = section.get("use_stable_emg")
    elif isinstance(section, SimpleNamespace):
        if hasattr(section, "emg_stable_mode"):
            raw = getattr(section, "emg_stable_mode")
        else:
            raw = getattr(section, "use_stable_emg", None)
    else:
        raise TypeError("Fitting configuration must be a mapping or namespace")
    if raw is None:
        return None
    return _coerce_optional_bool(raw)


def resolve_emg_stable_mode(
    *,
    config: Mapping[str, Any] | SimpleNamespace | None | object = _SENTINEL,
    override: bool | None | object = _SENTINEL,
    fallback: bool | object = _SENTINEL,
) -> bool:
    """Resolve the effective EMG stable mode respecting overrides and defaults."""

    cfg = _CONFIG_CACHE if config is _SENTINEL else cast(
        Mapping[str, Any] | SimpleNamespace | None, config
    )
    override_flag = _OVERRIDE if override is _SENTINEL else cast(bool | None, override)
    fallback_flag = _FALLBACK if fallback is _SENTINEL else bool(fallback)

    if override_flag is not None and override_flag is not _SENTINEL:
        return bool(override_flag)

    value = emg_stable_mode_from_config(cfg)
    if value is not None:
        return value

    return bool(fallback_flag)


def synchronize_emg_stable_mode(
    config: Mapping[str, Any] | SimpleNamespace | None | object = _SENTINEL,
    *,
    override: bool | None | object = _SENTINEL,
    fallback: bool | object = _SENTINEL,
) -> bool:
    """Persist EMG configuration state and update the core fitting module."""

    global _CONFIG_CACHE, _OVERRIDE, _FALLBACK

    if config is not _SENTINEL:
        _CONFIG_CACHE = config
    if override is not _SENTINEL:
        _OVERRIDE = cast(bool | None, override)
    if fallback is not _SENTINEL:
        _FALLBACK = bool(fallback)

    mode = resolve_emg_stable_mode()

    try:
        import fitting as _fitting  # type: ignore

        setattr(_fitting, "EMG_STABLE_MODE", mode)
    except ImportError:  # pragma: no cover - optional dependency
        pass

    return mode
