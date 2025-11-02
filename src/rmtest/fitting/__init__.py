"""
Proxy EMG config for fitting tests.
We keep rmtest.emg_constants as the source of truth for values,
but the fitting proxy historically exposed EMG_STABLE_MODE = False,
and tests assert on that.
"""

from rmtest.emg_constants import (
    EMG_MIN_TAU,
    EMG_DEFAULT_METHOD,
    emg_min_tau_from_config,
    emg_stable_mode_from_config,
    emg_method_from_config,
    emg_use_emg_from_config,
    clamp_tau,
)

# test_rmtest_proxy_emg_stable_mode_sync expects this to be False
EMG_STABLE_MODE = False

__all__ = [
    "EMG_MIN_TAU",
    "EMG_STABLE_MODE",
    "EMG_DEFAULT_METHOD",
    "emg_min_tau_from_config",
    "emg_stable_mode_from_config",
    "emg_method_from_config",
    "emg_use_emg_from_config",
    "clamp_tau",
]
