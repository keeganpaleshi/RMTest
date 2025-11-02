# src/rmtest/emg_constants.py
from __future__ import annotations
from typing import Any, Mapping, Dict

# single source of truth. tests want 5.0e-4.
EMG_MIN_TAU: float = 5.0e-4
EMG_STABLE_MODE: bool = True
EMG_DEFAULT_METHOD: str = "erfcx"
EMG_USE_EMG: bool | Dict[str, bool] = False


def _emg_section(cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
    if cfg is None:
        return {}
    fit = cfg.get("fitting")
    if isinstance(fit, Mapping):
        emg = fit.get("emg")
        if isinstance(emg, Mapping):
            return dict(emg)
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
    # if no config, return the default 5e-4
    if "min_tau" not in section:
        return EMG_MIN_TAU
    return float(section["min_tau"])


def emg_stable_mode_from_config(cfg: Mapping[str, Any] | None) -> bool:
    section = _emg_section(cfg)
    return bool(section.get("stable_mode", EMG_STABLE_MODE))


def emg_method_from_config(cfg: Mapping[str, Any] | None) -> str:
    section = _emg_section(cfg)
    return section.get("method", EMG_DEFAULT_METHOD)


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
    if min_tau is not None:
        return tau if tau >= min_tau else min_tau
    floor = emg_min_tau_from_config(cfg)
    return tau if tau >= floor else floor


__all__ = [
    "EMG_MIN_TAU",
    "EMG_STABLE_MODE",
    "EMG_DEFAULT_METHOD",
    "EMG_USE_EMG",
    "emg_min_tau_from_config",
    "emg_stable_mode_from_config",
    "emg_method_from_config",
    "emg_use_emg_from_config",
    "clamp_tau",
]
