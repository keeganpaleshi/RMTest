"""Helpers for keeping EMG configuration in sync for :mod:`rmtest`."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from types import SimpleNamespace
from typing import Any, Tuple

DEFAULT_EMG_STABLE_MODE = True

_LATEST_MODE: bool = DEFAULT_EMG_STABLE_MODE
_MODE_OVERRIDE: bool | None = None


def _coerce_bool(value: object | None) -> bool | None:
    """Coerce ``value`` to :class:`bool` when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "auto", "none", "null"}:
            return None
        if text in {"true", "yes", "on", "1"}:
            return True
        if text in {"false", "no", "off", "0"}:
            return False
    return bool(value)


def _get_attr(obj: object, name: str) -> object | None:
    if isinstance(obj, Mapping):
        return obj.get(name)
    if isinstance(obj, SimpleNamespace):
        return getattr(obj, name, None)
    return None


def resolve_emg_config(
    source: Mapping[str, Any] | SimpleNamespace | None,
    *,
    default: bool = DEFAULT_EMG_STABLE_MODE,
) -> Tuple[bool, bool]:
    """Return the ``(mode, use)`` tuple resolved from ``source``.

    ``mode`` is the preferred value for ``emg_stable_mode`` and ``use`` is the
    derived ``use_stable_emg`` flag. The precedence order is:

    1. Explicit ``emg_stable_mode`` value.
    2. Explicit ``use_stable_emg`` value.
    3. ``default`` when nothing is provided.

    Parameters
    ----------
    source:
        Mapping or namespace providing EMG configuration keys.
    default:
        Fallback value used when neither key is provided.
    """

    mode_raw = _get_attr(source, "emg_stable_mode")
    use_raw = _get_attr(source, "use_stable_emg")

    mode_pref = _coerce_bool(mode_raw)
    use_pref = _coerce_bool(use_raw)

    if mode_pref is None:
        if use_pref is not None:
            mode_pref = use_pref
        else:
            mode_pref = bool(default)

    if use_pref is None:
        use_pref = mode_pref

    return mode_pref, use_pref


def ensure_emg_config(
    fit_cfg: MutableMapping[str, Any] | None,
    *,
    default: bool = DEFAULT_EMG_STABLE_MODE,
) -> Tuple[bool, bool]:
    """Ensure ``fit_cfg`` contains consistent EMG settings.

    Parameters
    ----------
    fit_cfg:
        Mutable mapping representing the ``fitting`` configuration section. If
        ``None`` the mapping will not be mutated and the defaults are used.
    default:
        Fallback value when neither EMG key is present.

    Returns
    -------
    tuple
        ``(mode, use)`` booleans suitable for assigning to
        ``emg_stable_mode`` and ``use_stable_emg`` respectively.
    """

    mode, use = resolve_emg_config(fit_cfg, default=default)

    if isinstance(fit_cfg, MutableMapping):
        fit_cfg["emg_stable_mode"] = mode
        fit_cfg["use_stable_emg"] = use

    return mode, use


def _ensure_fit_section(
    cfg: Mapping[str, Any] | MutableMapping[str, Any] | SimpleNamespace | None,
) -> MutableMapping[str, Any] | None:
    if isinstance(cfg, MutableMapping):
        fit_cfg = cfg.get("fitting")
        if fit_cfg is None:
            fit_cfg = {}
            cfg["fitting"] = fit_cfg
        elif not isinstance(fit_cfg, MutableMapping):
            raise TypeError("'fitting' section must be a mapping if provided")
        return fit_cfg

    if isinstance(cfg, Mapping):
        fit_cfg = cfg.get("fitting")
        if isinstance(fit_cfg, MutableMapping):
            return fit_cfg
        return None

    if isinstance(cfg, SimpleNamespace):
        maybe = getattr(cfg, "fitting", None)
        if isinstance(maybe, MutableMapping):
            return maybe
        return None

    return None


def load_emg_config(
    cfg: Mapping[str, Any] | MutableMapping[str, Any] | SimpleNamespace | None,
    *,
    default: bool = DEFAULT_EMG_STABLE_MODE,
) -> bool:
    """Normalize EMG configuration from ``cfg`` and update module state.

    The ``cfg`` mapping is mutated in-place when possible. The return value is
    the resolved ``emg_stable_mode`` preference taking configuration and local
    overrides into account.
    """

    fit_cfg = _ensure_fit_section(cfg)
    mode, _ = ensure_emg_config(fit_cfg, default=default)
    set_default_mode(mode)
    return mode


def set_default_mode(value: bool) -> None:
    """Update the default EMG mode resolved from configuration."""

    global _LATEST_MODE
    _LATEST_MODE = bool(value)


def set_mode_override(value: bool | None) -> None:
    """Override the EMG mode regardless of configuration."""

    global _MODE_OVERRIDE, _LATEST_MODE

    if value is None:
        _MODE_OVERRIDE = None
        return

    coerced = bool(value)
    _MODE_OVERRIDE = coerced
    _LATEST_MODE = coerced


def clear_mode_override() -> None:
    """Reset any explicit EMG mode override."""

    set_mode_override(None)


def get_emg_stable_mode(default: bool | None = None) -> bool:
    """Return the effective EMG stable mode preference."""

    if _MODE_OVERRIDE is not None:
        return _MODE_OVERRIDE

    if default is not None:
        return bool(_LATEST_MODE if _LATEST_MODE is not None else default)

    return _LATEST_MODE


__all__ = [
    "DEFAULT_EMG_STABLE_MODE",
    "clear_mode_override",
    "ensure_emg_config",
    "get_emg_stable_mode",
    "load_emg_config",
    "resolve_emg_config",
    "set_default_mode",
    "set_mode_override",
]
