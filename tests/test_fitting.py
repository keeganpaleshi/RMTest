import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fitting import fit_decay


def simulate_decay(E_true, eff, T, n_events=1000):
    """Generate simple constant-rate decay times."""
    rate = E_true * eff
    n = np.random.poisson(rate * T)
    return np.sort(np.random.uniform(0, T, n))


def test_fit_decay_po214_only():
    # Simulate simple scenario
    T = 3600  # 1 hour
    t_half = 164e-6
    lambda_decay = np.log(2) / t_half
    E_true = 0.5  # decays/s
    eff = 0.4
    event_times = simulate_decay(E_true, eff, T)

    # Add some Po-218 artificially? Skip for Po-214-only test
    # Fit
    class DummyConfig:
        def __init__(self):
            self.fit_options = {
                "fit_po218_po214": False,
                "fix_B": False,
                "fix_N0": False,
                "baseline_range": None,
            }
            self.efficiency = {"eff_po214": eff, "eff_po218": 1.0}

    params_fit, _ = fit_decay(
        event_times,
        T,
        lambda_decay,
        eff,
        {"fit_options": DummyConfig().fit_options, "efficiency": {"eff_po214": eff}},
    )
    E_fit, N0_fit, B_fit = params_fit
    # Basic sanity: E_fit should be within factor of 2 of E_true (low stats)
    assert E_fit > 0
    assert abs(E_fit - E_true) / E_true < 1.0
