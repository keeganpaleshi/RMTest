"""Compatibility helpers for configuration loading in :mod:`rmtest`."""

from __future__ import annotations

from typing import Any, Mapping

from io_utils import load_config as _core_load_config


def load_config(config_path: str | Mapping[str, Any]) -> Mapping[str, Any]:
    """Load a configuration mapping and keep EMG preferences in sync."""

    cfg = _core_load_config(config_path)

    try:  # Ensure the compatibility layer tracks the resolved EMG mode.
        from .fitting import emg_config as _emg_config

        _emg_config.load_emg_config(cfg)
    except ImportError:  # pragma: no cover - package may be partially installed
        pass

    return cfg


__all__ = ["load_config"]
