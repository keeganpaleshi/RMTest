"""Central EMG/tau constants and configuration helpers.

This module is the SINGLE source of truth for EMG-related constants
and configuration defaults across the RMTest codebase.

All other modules should import from here rather than defining their own values.
"""

from __future__ import annotations

import os

# Single source of truth; tests expect 5e-4
_DEFAULT_EMG_MIN_TAU = 5e-4

# Allow override by env but never go below 5e-4
_env_val = os.environ.get("RMTEST_EMG_MIN_TAU")
if _env_val is not None:
    try:
        EMG_MIN_TAU = max(float(_env_val), _DEFAULT_EMG_MIN_TAU)
    except ValueError:
        EMG_MIN_TAU = _DEFAULT_EMG_MIN_TAU
else:
    EMG_MIN_TAU = _DEFAULT_EMG_MIN_TAU

# Tests expect the stable-mode proxy to match this
EMG_STABLE_MODE = True

# Keep the name they've been using
EMG_DEFAULT_METHOD = "erfcx"


def emg_min_tau_from_config(cfg: dict | None) -> float:
    """Extract minimum tau from config or return default.

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Minimum tau value, always >= EMG_MIN_TAU
    """
    if not cfg:
        return EMG_MIN_TAU
    return float(cfg.get("emg_tau_min", EMG_MIN_TAU))


def emg_stable_mode_from_config(cfg: dict | None) -> bool | str:
    """Extract EMG stable mode from config or return default.

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Stable mode setting (bool or string)
    """
    if not cfg:
        return EMG_STABLE_MODE
    return cfg.get("emg_stable_mode", EMG_STABLE_MODE)


def emg_method_from_config(cfg: dict | None) -> str:
    """Extract EMG method from config or return default.

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        EMG method name (e.g., "erfcx", "scipy_safe")
    """
    if not cfg:
        return EMG_DEFAULT_METHOD
    return cfg.get("emg_method", EMG_DEFAULT_METHOD)


def emg_use_emg_from_config(cfg: dict | None) -> bool:
    """Extract use_emg flag from config or return default.

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Whether to use EMG tails
    """
    if not cfg:
        return False
    return bool(cfg.get("use_emg", False))


def clamp_tau(tau: float | None, cfg: dict | None = None) -> float:
    """Clamp tau to the configured minimum floor.

    Args:
        tau: Tau value to clamp, may be None
        cfg: Configuration dictionary, may be None

    Returns:
        Clamped tau value, always >= floor
    """
    floor = emg_min_tau_from_config(cfg)
    if tau is None:
        return floor
    return max(tau, floor)


__all__ = [
    "EMG_STABLE_MODE",
    "EMG_MIN_TAU",
    "EMG_DEFAULT_METHOD",
    "emg_min_tau_from_config",
    "emg_stable_mode_from_config",
    "emg_method_from_config",
    "emg_use_emg_from_config",
    "clamp_tau",
]
