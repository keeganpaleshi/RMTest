import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fitting import fit_decay
from fitting import fit_spectrum


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
    res = fit_decay(
        times=event_times,
        priors={
            "eff": (eff, 0.0),
            "tau": (1.0 / lambda_decay, 0.0),
        },
        t0=0.0,
        t_end=T,
        flags={},
    )
    E_fit = res["E"]
    # Basic sanity: E_fit should be within factor of 2 of E_true (low stats)
    assert E_fit > 0
    assert abs(E_fit - E_true) / E_true < 1.0


def test_fit_decay_time_window_config():
    """Changing the energy window should alter the events passed to fit_decay."""
    # Two groups of events at different energies
    times = np.linspace(0, 10, 10)
    energies = np.array([7.65, 7.75, 7.85, 7.55, 7.70, 7.80, 7.40, 7.90, 7.60, 7.50])

    # Config window covering most events
    cfg = {"time_fit": {"window_Po214": [7.6, 7.9]}}
    w = cfg["time_fit"]["window_Po214"]
    mask = (energies >= w[0]) & (energies <= w[1])
    res_full = fit_decay(times[mask], {"eff": (1.0, 0.0)}, t0=0.0, t_end=10)
    count_full = mask.sum()

    # Narrower window -> fewer events
    cfg_narrow = {"time_fit": {"window_Po214": [7.7, 7.8]}}
    w2 = cfg_narrow["time_fit"]["window_Po214"]
    mask2 = (energies >= w2[0]) & (energies <= w2[1])
    res_narrow = fit_decay(times[mask2], {"eff": (1.0, 0.0)}, t0=0.0, t_end=10)
    count_narrow = mask2.sum()

    assert count_narrow < count_full
    assert res_narrow["E"] < res_full["E"]


def test_fit_spectrum_use_emg_flag():
    """Adding a tau prior when use_emg is True should not break the fit."""
    rng = np.random.default_rng(0)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 200),
        rng.normal(6.0, 0.05, 200),
        rng.normal(7.7, 0.05, 200),
    ])

    base_priors = {
        "sigma_E": (0.05, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (200, 20),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (200, 20),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (200, 20),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    # Without EMG tail
    out_no_emg = fit_spectrum(energies, dict(base_priors))

    # With EMG tail for Po218
    priors_emg = dict(base_priors)
    priors_emg["tau_Po218"] = (5.0, 1.0)
    out_emg = fit_spectrum(energies, priors_emg)

    # The EMG fit should return a tau parameter and modify the peak shape
    assert "tau_Po218" in out_emg
    assert out_emg["tau_Po218"] != 0
    assert abs(out_emg["S_Po218"] - out_no_emg["S_Po218"]) > 1e-3


def test_fit_spectrum_fixed_parameter_bounds():
    """Fixing a parameter should not trigger a bound error."""
    rng = np.random.default_rng(1)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 100),
        rng.normal(6.0, 0.05, 100),
        rng.normal(7.7, 0.05, 100),
    ])

    priors = {
        "sigma_E": (0.05, 0.01),
        "mu_Po210": (5.3, 0.0),
        "S_Po210": (100, 10),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (100, 10),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (100, 10),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    out = fit_spectrum(energies, priors, flags={"fix_mu_Po210": True})
    assert "mu_Po210" in out


def test_fit_spectrum_custom_bins_and_edges():
    """Providing custom binning should not break the fit."""
    rng = np.random.default_rng(2)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 150),
        rng.normal(6.0, 0.05, 150),
        rng.normal(7.7, 0.05, 150),
    ])

    priors = {
        "sigma_E": (0.05, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (150, 15),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (150, 15),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (150, 15),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    # Using integer number of bins
    out_bins = fit_spectrum(energies, priors, bins=30)
    assert "sigma_E" in out_bins

    # Using explicit bin edges
    edges = np.linspace(5.0, 8.0, 25)
    out_edges = fit_spectrum(energies, priors, bin_edges=edges)
    assert "sigma_E" in out_edges


def test_fit_spectrum_custom_bounds():
    """User-provided parameter bounds should constrain the fit."""
    rng = np.random.default_rng(3)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 150),
        rng.normal(6.0, 0.05, 150),
        rng.normal(7.7, 0.05, 150),
    ])

    priors = {
        "sigma_E": (0.05, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (150, 15),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (150, 15),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (150, 15),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    bounds = {"mu_Po218": (5.9, 6.1)}
    out = fit_spectrum(energies, priors, bounds=bounds)
    assert 5.9 <= out["mu_Po218"] <= 6.1


def test_fit_spectrum_bounds_clip():
    """Starting value outside the bound should be clipped before fitting."""
    rng = np.random.default_rng(4)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 120),
        rng.normal(6.0, 0.05, 120),
        rng.normal(7.7, 0.05, 120),
    ])

    priors = {
        "sigma_E": (0.05, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (120, 12),
        "mu_Po218": (5.5, 0.1),  # outside the provided bound
        "S_Po218": (120, 12),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (120, 12),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    bounds = {"mu_Po218": (5.9, 6.1)}
    lo, hi = bounds["mu_Po218"]
    mu_clipped = np.clip(priors["mu_Po218"][0], lo, hi)
    priors["mu_Po218"] = (mu_clipped, priors["mu_Po218"][1])

    out = fit_spectrum(energies, priors, bounds=bounds)
    assert lo <= out["mu_Po218"] <= hi
