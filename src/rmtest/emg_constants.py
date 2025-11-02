"""Central EMG/tau constants and configuration helpers.

This module provides a single source of truth for EMG-related constants
and configuration defaults across the RMTest codebase.
"""

from __future__ import annotations

from typing import Any, Mapping

# EMG stable mode: use numerically stable erfcx implementation by default
EMG_STABLE_MODE: bool = True

# Minimum tau value (in seconds or ADC equivalent)
# This is the hard floor below which tau values will be clamped
EMG_MIN_TAU: float = 5e-4

# Default EMG numerical method
# "erfcx" = stable implementation using erfcx function
# "direct" = legacy direct calculation
EMG_DEFAULT_METHOD: str = "erfcx"


def emg_min_tau_from_config(cfg: Mapping[str, Any] | None) -> float:
    """Extract the minimum tau value from configuration.

    Resolution order:
    1. cfg["fitting"]["emg"]["min_tau"] if present
    2. EMG_MIN_TAU default

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Minimum tau value as a float
    """
    if cfg is None:
        return EMG_MIN_TAU

    return float(
        cfg.get("fitting", {})
        .get("emg", {})
        .get("min_tau", EMG_MIN_TAU)
    )


def emg_stable_mode_from_config(cfg: Mapping[str, Any] | None) -> bool:
    """Extract the EMG stable mode setting from configuration.

    Resolution order:
    1. cfg["fitting"]["emg"]["stable_mode"] if present
    2. EMG_STABLE_MODE default

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Whether to use stable EMG mode as a boolean
    """
    if cfg is None:
        return EMG_STABLE_MODE

    return bool(
        cfg.get("fitting", {})
        .get("emg", {})
        .get("stable_mode", EMG_STABLE_MODE)
    )


def emg_method_from_config(cfg: Mapping[str, Any] | None) -> str:
    """Extract the EMG numerical method from configuration.

    Resolution order:
    1. cfg["fitting"]["emg"]["method"] if present
    2. EMG_DEFAULT_METHOD default

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        EMG method name as a string ("erfcx" or "direct")
    """
    if cfg is None:
        return EMG_DEFAULT_METHOD

    method = cfg.get("fitting", {}).get("emg", {}).get("method", EMG_DEFAULT_METHOD)
    return str(method).lower()


def emg_use_emg_from_config(cfg: Mapping[str, Any] | None) -> bool | dict[str, bool]:
    """Extract the use_emg setting from configuration.

    Resolution order for per-isotope determination:
    1. If tau_{iso}_prior_mean exists, EMG is ON for that isotope
    2. If cfg["fitting"]["emg"]["use_emg"] is a mapping, use that value
    3. If cfg["fitting"]["emg"]["use_emg"] is a bool, use that
    4. Default is False (EMG OFF)

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Either a boolean (global setting) or dict mapping isotope -> bool
    """
    if cfg is None:
        return False

    use_emg = cfg.get("fitting", {}).get("emg", {}).get("use_emg", False)

    # Return as-is if it's already a mapping or bool
    if isinstance(use_emg, bool):
        return use_emg
    elif isinstance(use_emg, Mapping):
        return dict(use_emg)
    else:
        # Coerce to bool if it's some other type
        return bool(use_emg)


def clamp_tau(
    tau: float,
    cfg: Mapping[str, Any] | None = None,
    min_tau: float | None = None,
) -> float:
    """Clamp tau to the minimum allowed value.

    Args:
        tau: The tau value to clamp
        cfg: Configuration dictionary, may be None
        min_tau: Override minimum tau (if None, uses config or default)

    Returns:
        Clamped tau value
    """
    if min_tau is None:
        min_tau = emg_min_tau_from_config(cfg)

    return max(tau, min_tau)


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
