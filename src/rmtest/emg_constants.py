"""Central EMG/tau constants and configuration helpers.

This module is the SINGLE source of truth for EMG-related constants
and configuration defaults across the RMTest codebase.

All other modules should import from here rather than defining their own values.
"""

from __future__ import annotations

from typing import Any, Mapping, Dict

# tests expect 5e-4 as the built-in floor
EMG_MIN_TAU: float = 5.0e-4

# tests expect this to be ON by default
EMG_STABLE_MODE: bool = True

# tests import this exact name
EMG_DEFAULT_METHOD: str = "erfcx"
# keep old alias if other code used it
EMG_METHOD = EMG_DEFAULT_METHOD

# can be bool or per-isotope dict
EMG_USE_EMG: bool | Dict[str, bool] = False


def _emg_section(cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
    if cfg is None:
        return {}
    fit = cfg.get("fitting")
    if isinstance(fit, Mapping):
        # new style
        emg = fit.get("emg")
        if isinstance(emg, Mapping):
            return dict(emg)
        # legacy keys
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
    section = _emg_section(cfg)
    value = section.get("min_tau", EMG_MIN_TAU)
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = EMG_MIN_TAU
    # hard floor: never go below 5e-4 from config
    return v if v >= 5.0e-4 else 5.0e-4


def emg_stable_mode_from_config(cfg: Mapping[str, Any] | None) -> bool:
    section = _emg_section(cfg)
    return bool(section.get("stable_mode", EMG_STABLE_MODE))


def emg_method_from_config(cfg: Mapping[str, Any] | None) -> str:
    section = _emg_section(cfg)
    method = section.get("method", EMG_DEFAULT_METHOD)
    # normalize to what your code uses
    if method in ("erfcx", "erfcx_exact"):
        return "erfcx"
    if method in ("direct", "scipy_safe", "legacy", "exponnorm"):
        return "direct"
    return EMG_DEFAULT_METHOD


def emg_use_emg_from_config(cfg: Mapping[str, Any] | None):
    section = _emg_section(cfg)
    if "use_emg" in section:
        return section["use_emg"]
    return EMG_USE_EMG


def clamp_tau(
    tau: float,
    cfg: Mapping[str, Any] | None = None,
    *,
    min_tau: float | None = None,
) -> float:
    # THIS is what your failing tests want:
    # explicit kw arg wins even if it is 1e-4 < global 5e-4
    if min_tau is not None:
        return tau if tau >= min_tau else min_tau
    floor = emg_min_tau_from_config(cfg)
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
