"""Centralized defaults and helpers for EMG fitting configuration.

This module holds the single source of truth for Exponentially Modified
Gaussian (EMG) configuration defaults used throughout the project.

Constants
---------
EMG_MIN_TAU
    Minimum allowable EMG decay constant (``tau``) in seconds for numerical
    stability.
EMG_STABLE_MODE
    Whether to use the numerically stable EMG implementation by default.
EMG_DEFAULT_METHOD
    Default EMG evaluation method identifier used by the fitting routines.
EMG_USE_EMG
    Whether EMG should be used at all; can be a simple boolean or a mapping of
    dataset names to booleans to enable EMG selectively.
"""

from __future__ import annotations
from typing import Any, Mapping, Dict

EMG_MIN_TAU: float = 5.0e-4
EMG_STABLE_MODE: bool = True
EMG_DEFAULT_METHOD: str = "erfcx"
EMG_USE_EMG: bool | Dict[str, bool] = False


def _emg_section(cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Extract the EMG subsection from a configuration mapping.

    The configuration is expected to have a top-level ``"fitting"`` mapping.
    Within that scope, EMG values may be nested under an ``"emg"`` mapping or
    provided with legacy keys (e.g., ``"emg_tau_min"``, ``"emg_method"``).
    Values are copied into a new dictionary so callers can mutate the result
    safely.

    Parameters
    ----------
    cfg
        The overall configuration mapping or ``None``.

    Returns
    -------
    dict
        A shallow dictionary containing EMG configuration fields with keys
        such as ``"min_tau"``, ``"stable_mode"``, ``"method"``, or
        ``"use_emg"``.
    """
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
    """Return the minimum EMG ``tau`` (seconds) from configuration.

    The function reads ``min_tau`` from the ``fitting.emg`` section (or legacy
    ``emg_tau_min``) if provided; otherwise, it falls back to
    :data:`EMG_MIN_TAU`.
    """
    section = _emg_section(cfg)
    # if no config, return the default 5e-4
    if "min_tau" not in section:
        return EMG_MIN_TAU
    return float(section["min_tau"])


def emg_stable_mode_from_config(cfg: Mapping[str, Any] | None) -> bool:
    """Return whether to use the numerically stable EMG implementation.

    Reads ``stable_mode`` from ``fitting.emg`` (or legacy ``use_stable_emg`` /
    ``emg_stable_mode``) and defaults to :data:`EMG_STABLE_MODE` when absent.
    """
    section = _emg_section(cfg)
    return bool(section.get("stable_mode", EMG_STABLE_MODE))


def emg_method_from_config(cfg: Mapping[str, Any] | None) -> str:
    """Return the EMG evaluation method identifier from configuration.

    The method is read from ``fitting.emg.method`` (or legacy ``emg_method``)
    and defaults to :data:`EMG_DEFAULT_METHOD`.
    """
    section = _emg_section(cfg)
    return section.get("method", EMG_DEFAULT_METHOD)


def emg_use_emg_from_config(cfg: Mapping[str, Any] | None):
    """Return whether EMG fitting is enabled.

    This value may be a boolean for global control or a mapping of dataset
    names to booleans to enable EMG selectively. It is retrieved from
    ``fitting.emg.use_emg`` (or legacy ``emg_use_emg``) and defaults to
    :data:`EMG_USE_EMG`.
    """
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
    """Clamp ``tau`` to a configured minimum value.

    Parameters
    ----------
    tau
        The proposed EMG decay constant in seconds.
    cfg
        Optional configuration mapping used to derive the minimum ``tau``
        through :func:`emg_min_tau_from_config` when ``min_tau`` is not
        provided.
    min_tau
        An explicit lower bound in seconds. When given, configuration values
        are ignored.

    Returns
    -------
    float
        ``tau`` if it is above the applicable minimum; otherwise, the minimum
        value in seconds.
    """
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
