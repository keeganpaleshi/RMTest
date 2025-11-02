"""Centralized EMG/config helpers for fitting.

Important
---------
Do NOT import the top-level `fitting.py` from here. That file imports us,
so importing it here creates a circular import during test collection.

This module only exports EMG-related configuration helpers. If you need
access to FitResult or other fitting module members, import them directly
from the top-level fitting module, not through this package.
"""

from __future__ import annotations

from .emg_config import (
    get_emg_stable_mode,
    resolve_emg_mode_preference,
    set_emg_mode_from_config,
    set_emg_mode_override,
    reset_emg_mode_preferences,
)
from .emg_utils import EMGTailSpec, resolve_emg_usage

__all__ = [
    "get_emg_stable_mode",
    "resolve_emg_mode_preference",
    "set_emg_mode_from_config",
    "set_emg_mode_override",
    "reset_emg_mode_preferences",
    "EMGTailSpec",
    "resolve_emg_usage",
]


def get_FitResult():
    """Lazy import of FitResult to avoid circular imports.

    Returns
    -------
    type
        The FitResult class from the top-level fitting module.

    Notes
    -----
    This function imports fitting.FitResult lazily at call time rather than
    at module import time to avoid circular import issues.
    """
    from fitting import FitResult
    return FitResult
