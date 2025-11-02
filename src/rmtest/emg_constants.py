"""Shared constants and configuration helpers for EMG handling."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

EMG_STABLE_MODE: bool = True
EMG_MIN_TAU: float = 5e-4  # seconds or ADC equiv, match your current unit
EMG_DEFAULT_METHOD: str = "erfcx"  # or "direct" if user disables stable path


def _get_section(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, {})
    if isinstance(source, SimpleNamespace):
        return getattr(source, key, SimpleNamespace())
    return getattr(source, key, {})


def emg_min_tau_from_config(cfg: Mapping[str, Any] | SimpleNamespace | None) -> float:
    """Extract the EMG tau floor from ``cfg`` falling back to :data:`EMG_MIN_TAU`."""

    if cfg is None:
        return float(EMG_MIN_TAU)
    emg_cfg = _get_section(_get_section(cfg, "fitting"), "emg")
    if isinstance(emg_cfg, Mapping):
        value = emg_cfg.get("min_tau", EMG_MIN_TAU)
    else:
        value = getattr(emg_cfg, "min_tau", EMG_MIN_TAU)
    return float(value)


def emg_stable_mode_from_config(
    cfg: Mapping[str, Any] | SimpleNamespace | None,
) -> bool:
    """Extract the EMG stable mode flag from ``cfg`` using :data:`EMG_STABLE_MODE`."""

    if cfg is None:
        return bool(EMG_STABLE_MODE)
    emg_cfg = _get_section(_get_section(cfg, "fitting"), "emg")
    if isinstance(emg_cfg, Mapping):
        value = emg_cfg.get("stable_mode", EMG_STABLE_MODE)
    else:
        value = getattr(emg_cfg, "stable_mode", EMG_STABLE_MODE)
    return bool(value)


__all__ = [
    "EMG_STABLE_MODE",
    "EMG_MIN_TAU",
    "EMG_DEFAULT_METHOD",
    "emg_min_tau_from_config",
    "emg_stable_mode_from_config",
]
