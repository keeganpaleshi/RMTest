"""Configuration helpers for the :mod:`rmtest` compatibility package."""

from __future__ import annotations

from typing import Any

import io_utils

from .fitting.emg_config import (
    DEFAULT_EMG_STABLE_MODE,
    EmgPreferences,
    apply_emg_preferences,
    resolve_emg_preferences,
)

__all__ = [
    "DEFAULT_EMG_STABLE_MODE",
    "EmgPreferences",
    "apply_emg_preferences",
    "load_config",
    "resolve_emg_preferences",
]


def load_config(config: Any):
    """Load a configuration using :mod:`io_utils` and sync EMG preferences."""

    cfg = io_utils.load_config(config)
    apply_emg_preferences(cfg, default_mode=DEFAULT_EMG_STABLE_MODE)
    return cfg
