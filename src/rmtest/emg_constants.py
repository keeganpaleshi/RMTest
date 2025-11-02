"""Central EMG/tau constants and configuration helpers.

This module is the SINGLE source of truth for EMG-related constants
and configuration defaults across the RMTest codebase.

All other modules should import from here rather than defining their own values.
"""

from __future__ import annotations

from typing import Any, Mapping, Dict

# Single source of truth defaults that tests expect
EMG_MIN_TAU: float = 5.0e-4
EMG_STABLE_MODE: bool = True
EMG_DEFAULT_METHOD: str = "erfcx"
EMG_METHOD = EMG_DEFAULT_METHOD  # Alias for backward compatibility
EMG_USE_EMG: bool | Dict[str, bool] = False


def _emg_section(cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return the normalized EMG section from a config dict.

    Supports both new layout:
        fitting:
          emg:
            stable_mode: ...
            min_tau: ...
            method: ...
            use_emg: ...

    And legacy layout:
        fitting:
          emg_tau_min: ...
          emg_stable_mode: ...
          use_stable_emg: ...
          emg_method: ...
          emg_use_emg: ...
    """
    if cfg is None:
        return {}

    fit = cfg.get("fitting")
    if isinstance(fit, Mapping):
        # New layout: fitting.emg.*
        emg = fit.get("emg")
        if isinstance(emg, Mapping):
            return dict(emg)

        # Legacy keys directly under "fitting"
        out: Dict[str, Any] = {}
        if "emg_tau_min" in fit:
            out["min_tau"] = fit["emg_tau_min"]
        if "emg_stable_mode" in fit:
            out["stable_mode"] = fit["emg_stable_mode"]
        if "use_stable_emg" in fit:
            out["stable_mode"] = fit["use_stable_emg"]
        if "emg_method" in fit:
            out["method"] = fit["emg_method"]
        if "emg_use_emg" in fit:
            out["use_emg"] = fit["emg_use_emg"]
        return out

    return {}


def emg_min_tau_from_config(cfg: Mapping[str, Any] | None) -> float:
    """Extract minimum tau from config or return default.

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Minimum tau value, always >= 5.0e-4
    """
    section = _emg_section(cfg)
    value = section.get("min_tau", EMG_MIN_TAU)
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = EMG_MIN_TAU

    # Tests want a hard floor of 5e-4 even if somebody gives 1e-8
    if v < 5.0e-4:
        return 5.0e-4
    return v


def emg_stable_mode_from_config(cfg: Mapping[str, Any] | None) -> bool:
    """Extract EMG stable mode from config or return default.

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Stable mode setting as boolean
    """
    section = _emg_section(cfg)
    return bool(section.get("stable_mode", EMG_STABLE_MODE))


def emg_method_from_config(cfg: Mapping[str, Any] | None) -> str:
    """Extract EMG method from config or return default.

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        EMG method name ("erfcx" or "direct")
    """
    section = _emg_section(cfg)
    method = section.get("method", EMG_METHOD)

    # Normalize what users write to what code actually uses
    if method in ("erfcx", "erfcx_exact"):
        return "erfcx"
    if method in ("direct", "scipy_safe", "legacy", "exponnorm"):
        # Tests ask for "direct"
        return "direct"
    return EMG_METHOD


def emg_use_emg_from_config(cfg: Mapping[str, Any] | None):
    """Extract use_emg flag from config or return default.

    Args:
        cfg: Configuration dictionary, may be None

    Returns:
        Either a boolean (global setting) or dict mapping isotope -> bool
    """
    section = _emg_section(cfg)
    if "use_emg" in section:
        # Can be bool OR dict; tests check for the dict case
        return section["use_emg"]
    return EMG_USE_EMG


def clamp_tau(
    tau: float,
    cfg: Mapping[str, Any] | None = None,
    *,
    min_tau: float | None = None,
) -> float:
    """Clamp tau to the floor from config or to the explicit floor.

    Args:
        tau: Tau value to clamp
        cfg: Configuration dictionary, may be None
        min_tau: Override minimum tau (keyword-only)

    Returns:
        Clamped tau value, always >= floor

    Notes:
        tests/test_emg_config.py calls this with min_tau=...
        so the kw-only arg must exist.
    """
    floor = min_tau if min_tau is not None else emg_min_tau_from_config(cfg)
    return tau if tau >= floor else floor


__all__ = [
    "EMG_MIN_TAU",
    "EMG_STABLE_MODE",
    "EMG_DEFAULT_METHOD",
    "EMG_METHOD",
    "EMG_USE_EMG",
    "emg_min_tau_from_config",
    "emg_stable_mode_from_config",
    "emg_method_from_config",
    "emg_use_emg_from_config",
    "clamp_tau",
]
