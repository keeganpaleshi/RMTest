import numpy as np
import pytest
from fitting import fit_decay


def simulate_decay(E_true, N0_true, B_true, lambda_decay, eff, T, n_events=1000):
    """Simulate Po-214 event times via thinning of inhomogeneous Poisson process."""
    # Use simple rejection sampling: find max rate
    t = 0.0
    times = []
    dt = T / n_events
    t_vals = np.linspace(0, T, n_events)
    max_rate = eff * (N0_true * lambda_decay + E_true)
    while t < T:
        t += np.random.exponential(1.0 / max_rate)
        if t >= T:
            break
        # Compute actual rate
        actual = (
            eff
            * (
                N0_true * lambda_decay * np.exp(-lambda_decay * t)
                + E_true * (1 - np.exp(-lambda_decay * t))
            )
            + B_true
        )
        if np.random.rand() * max_rate < actual:
            times.append(t)
    return np.array(times)


def test_fit_decay_po214_only():
    # Simulate simple scenario
    T = 3600  # 1 hour
    t_half = 164e-6
    lambda_decay = np.log(2) / t_half
    E_true = 0.5  # decays/s
    N0_true = 1000
    B_true = 0.01
    eff = 0.4
    event_times = simulate_decay(E_true, N0_true, B_true, lambda_decay, eff, T)

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
