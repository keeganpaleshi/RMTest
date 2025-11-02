"""rmtest package"""

from .emg_constants import (
    EMG_STABLE_MODE,
    EMG_MIN_TAU,
    EMG_DEFAULT_METHOD,
    EMG_METHOD,
    emg_min_tau_from_config,
    emg_stable_mode_from_config,
    emg_method_from_config,
    emg_use_emg_from_config,
    clamp_tau,
)

__all__ = [
    "EMG_STABLE_MODE",
    "EMG_MIN_TAU",
    "EMG_DEFAULT_METHOD",
    "EMG_METHOD",
    "emg_min_tau_from_config",
    "emg_stable_mode_from_config",
    "emg_method_from_config",
    "emg_use_emg_from_config",
    "clamp_tau",
]
