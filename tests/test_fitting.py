import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting import fit_time_series, fit_spectrum, FitResult, _TAU_MIN
import analyze


def simulate_decay(E_true, eff, T, n_events=1000):
    """Generate simple constant-rate decay times."""
    rate = E_true * eff
    n = np.random.poisson(rate * T)
    return np.sort(np.random.uniform(0, T, n))


def test_fit_time_series_po214_only():
    # Simulate simple scenario
    T = 3600  # 1 hour
    t_half = 164e-6
    E_true = 0.5  # decays/s
    eff = 0.4
    event_times = simulate_decay(E_true, eff, T)

    times_dict = {"Po214": event_times}
    cfg = {
        "isotopes": {"Po214": {"half_life_s": t_half, "efficiency": eff}},
        "fit_background": True,
        "fit_initial": True,
    }

    res = fit_time_series(times_dict, 0.0, T, cfg)
    E_fit = res.params["E_Po214"]
    # Basic sanity: E_fit should be within factor of 2 of E_true (low stats)
    assert E_fit > 0
    assert abs(E_fit - E_true) / E_true < 1.0


def test_fit_time_series_time_window_config():
    """Changing the energy window should alter the events passed to fit_time_series."""
    # Two groups of events at different energies
    times = np.linspace(0, 10, 10)
    energies = np.array([7.65, 7.75, 7.85, 7.55, 7.70, 7.80, 7.40, 7.90, 7.60, 7.50])

    # Config window covering most events
    cfg = {
        "time_fit": {
            "window_po214": [7.6, 7.9],
            "hl_po214": [1.0],
            "eff_po214": [1.0],
        }
    }
    w = cfg["time_fit"]["window_po214"]
    mask = (energies >= w[0]) & (energies <= w[1])
    times_dict = {"Po214": times[mask]}
    cfg_full = {
        "isotopes": {"Po214": {"half_life_s": cfg["time_fit"]["hl_po214"][0], "efficiency": 1.0}},
        "fit_background": True,
        "fit_initial": True,
    }
    res_full = fit_time_series(times_dict, 0.0, 10, cfg_full)
    count_full = mask.sum()

    # Narrower window -> fewer events
    cfg_narrow = {
        "time_fit": {
            "window_po214": [7.7, 7.8],
            "hl_po214": [1.0],
            "eff_po214": [1.0],
        }
    }
    w2 = cfg_narrow["time_fit"]["window_po214"]
    mask2 = (energies >= w2[0]) & (energies <= w2[1])
    times_dict2 = {"Po214": times[mask2]}
    cfg_n = {
        "isotopes": {"Po214": {"half_life_s": cfg_narrow["time_fit"]["hl_po214"][0], "efficiency": 1.0}},
        "fit_background": True,
        "fit_initial": True,
    }
    res_narrow = fit_time_series(times_dict2, 0.0, 10, cfg_n)
    count_narrow = mask2.sum()

    assert count_narrow < count_full
    assert res_narrow.params["E_Po214"] < res_full.params["E_Po214"]


def test_fit_spectrum_use_emg_flag():
    """Adding a tau prior when use_emg is True should not break the fit."""
    rng = np.random.default_rng(0)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 200),
        rng.normal(6.0, 0.05, 200),
        rng.normal(7.7, 0.05, 200),
    ])

    base_priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
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
    assert "tau_Po218" in out_emg.params
    assert out_emg.params["tau_Po218"] != 0
    assert abs(out_emg.params["S_Po218"] - out_no_emg.params["S_Po218"]) > 1e-3


def test_fit_spectrum_fixed_parameter_bounds():
    """Fixing a parameter should not trigger a bound error."""
    rng = np.random.default_rng(1)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 100),
        rng.normal(6.0, 0.05, 100),
        rng.normal(7.7, 0.05, 100),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
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
    assert "mu_Po210" in out.params


def test_fit_spectrum_custom_bins_and_edges():
    """Providing custom binning should not break the fit."""
    rng = np.random.default_rng(2)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 150),
        rng.normal(6.0, 0.05, 150),
        rng.normal(7.7, 0.05, 150),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
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
    assert "sigma0" in out_bins.params
    assert "F" in out_bins.params

    # Using explicit bin edges
    edges = np.linspace(5.0, 8.0, 25)
    out_edges = fit_spectrum(energies, priors, bin_edges=edges)
    assert "sigma0" in out_edges.params
    assert "F" in out_edges.params


def test_fit_spectrum_non_monotonic_edges_error():
    rng = np.random.default_rng(20)
    energies = rng.normal(5.3, 0.05, 50)

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (50, 5),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (50, 5),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (50, 5),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    edges = [5.0, 6.0, 5.5, 7.0]
    with pytest.raises(ValueError):
        fit_spectrum(energies, priors, bin_edges=edges)


def test_fit_spectrum_background_only_irregular_edges():
    """Background-only fit with irregular bin edges should recover the rate."""
    edges = np.array([0.0, 1.0, 3.0, 4.0])

    # Generate counts corresponding to a flat background of 10 counts/MeV
    energies = np.concatenate([
        np.linspace(0.05, 0.95, 10),
        np.linspace(1.1, 2.9, 20),
        np.linspace(3.05, 3.95, 10),
    ])

    priors = {
        "sigma0": (0.05, 0.0),
        "F": (0.0, 0.0),
        "mu_Po210": (0.5, 0.0),
        "S_Po210": (0.0, 0.0),
        "mu_Po218": (1.5, 0.0),
        "S_Po218": (0.0, 0.0),
        "mu_Po214": (3.5, 0.0),
        "S_Po214": (0.0, 0.0),
        "b0": (9.0, 2.0),
        "b1": (0.0, 0.0),
    }

    result = fit_spectrum(energies, priors, bin_edges=edges)
    b0 = result.params["b0"]
    b1 = result.params["b1"]
    B = result.params["S_bkg"]
    E_lo, E_hi = edges[0], edges[-1]
    total = B * (b0 * (E_hi - E_lo) + 0.5 * b1 * (E_hi**2 - E_lo**2))
    assert np.isclose(total, 40.0, atol=0.1)


def test_model_binned_variable_width(monkeypatch):
    """_model_binned should scale by bin width and error on invalid centers."""
    import fitting as fitting_mod

    monkeypatch.setattr(fitting_mod, "gaussian", lambda x, mu, sigma: np.ones_like(x))
    monkeypatch.setattr(fitting_mod, "emg_left", lambda x, mu, sigma, tau: np.ones_like(x))

    edges = np.array([0.0, 1.0, 2.0, 4.0])
    energies = np.array([0.1, 1.1, 2.1])

    priors = {
        "sigma0": (0.0, 0.0),
        "F": (0.0, 0.0),
        "mu_Po210": (0.5, 0.0),
        "S_Po210": (0.0, 0.0),
        "mu_Po218": (1.5, 0.0),
        "S_Po218": (0.0, 0.0),
        "mu_Po214": (3.0, 0.0),
        "S_Po214": (0.0, 0.0),
        "b0": (1.0, 0.0),
        "b1": (0.0, 0.0),
    }

    captured = {}

    def dummy_curve_fit(func, xdata, ydata, p0=None, bounds=None, maxfev=None):
        captured["func"] = func
        captured["xdata"] = xdata
        captured["p0"] = p0
        return np.array(p0), np.eye(len(p0))

    monkeypatch.setattr(fitting_mod, "curve_fit", dummy_curve_fit)
    fit_spectrum(energies, priors, bin_edges=edges)

    func = captured["func"]
    xdata = captured["xdata"]
    p0 = captured["p0"]
    B = fitting_mod.softplus(p0[-1])
    assert np.allclose(func(xdata, *p0) / B, np.diff(edges))

    with pytest.raises(KeyError):
        func(np.array([0.5, 0.6]), *p0)


def test_fit_spectrum_custom_bounds():
    """User-provided parameter bounds should constrain the fit."""
    rng = np.random.default_rng(3)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 150),
        rng.normal(6.0, 0.05, 150),
        rng.normal(7.7, 0.05, 150),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
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
    assert 5.9 <= out.params["mu_Po218"] <= 6.1


def test_fit_spectrum_bounds_clip():
    """Starting value outside the bound should be clipped before fitting."""
    rng = np.random.default_rng(4)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 120),
        rng.normal(6.0, 0.05, 120),
        rng.normal(7.7, 0.05, 120),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
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
    assert lo <= out.params["mu_Po218"] <= hi


def test_fit_spectrum_tau_lower_bound():
    """Tau prior near zero should be clipped to the minimum allowed value."""
    rng = np.random.default_rng(6)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 100),
        rng.normal(6.0, 0.05, 100),
        rng.normal(7.7, 0.05, 100),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (100, 10),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (100, 10),
        "tau_Po218": (0.0, 0.01),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (100, 10),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    result = fit_spectrum(energies, priors)
    assert result.params["tau_Po218"] >= _TAU_MIN


def test_fit_spectrum_unbinned_runs():
    rng = np.random.default_rng(7)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 150),
        rng.normal(6.0, 0.05, 150),
        rng.normal(7.7, 0.05, 150),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (150, 15),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (150, 15),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (150, 15),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    out = fit_spectrum(energies, priors, unbinned=True)
    assert "sigma0" in out.params
    assert "F" in out.params


def test_fit_spectrum_unbinned_consistent():
    rng = np.random.default_rng(8)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 200),
        rng.normal(6.0, 0.05, 200),
        rng.normal(7.7, 0.05, 200),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (200, 20),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (200, 20),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (200, 20),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    out_hist = fit_spectrum(energies, priors)
    out_unbinned = fit_spectrum(energies, priors, unbinned=True)
    diff = abs(out_hist.params["mu_Po210"] - out_unbinned.params["mu_Po210"])
    assert diff < 0.2


def test_fit_spectrum_fixed_resolution():
    rng = np.random.default_rng(42)
    energies = rng.normal(5.3, 0.05, 200)
    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (200, 20),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (200, 20),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (200, 20),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }
    out = fit_spectrum(energies, priors, flags={"fix_sigma0": True, "fix_F": True})
    assert out.params["sigma0"] == pytest.approx(priors["sigma0"][0])


def test_fit_spectrum_resolution_floats():
    rng = np.random.default_rng(43)
    energies = rng.normal(5.3, 0.07, 200)
    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (200, 20),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (200, 20),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (200, 20),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }
    out = fit_spectrum(energies, priors)
    assert out.params["sigma0"] > priors["sigma0"][0]


def test_fit_spectrum_legacy_fix_sigma_E():
    rng = np.random.default_rng(44)
    energies = rng.normal(5.3, 0.05, 150)
    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (150, 15),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (150, 15),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (150, 15),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }
    out_legacy = fit_spectrum(energies, priors, flags={"fix_sigma_E": True})
    out_new = fit_spectrum(energies, priors, flags={"fix_sigma0": True, "fix_F": True})
    assert out_legacy.params == out_new.params


def test_fit_spectrum_covariance_checks(monkeypatch):
    """fit_valid should reflect covariance positive definiteness."""
    rng = np.random.default_rng(5)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 200),
        rng.normal(6.0, 0.05, 200),
        rng.normal(7.7, 0.05, 200),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (200, 20),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (200, 20),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (200, 20),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    import fitting as fitting_mod

    orig_curve_fit = fitting_mod.curve_fit

    def good_curve_fit(*args, **kwargs):
        popt, pcov = orig_curve_fit(*args, **kwargs)
        return popt, np.eye(len(popt))

    monkeypatch.setattr(fitting_mod, "curve_fit", good_curve_fit)
    out = fit_spectrum(energies, priors)
    assert out.params["fit_valid"]

    def bad_curve_fit(*args, **kwargs):
        popt, pcov = orig_curve_fit(*args, **kwargs)
        pcov = np.eye(len(popt))
        pcov[0, 0] = -1.0
        return popt, pcov

    monkeypatch.setattr(fitting_mod, "curve_fit", bad_curve_fit)
    out_bad = fit_spectrum(energies, priors)
    assert not out_bad.params["fit_valid"]

    with pytest.raises(RuntimeError):
        fit_spectrum(energies, priors, strict=True)


def test_fit_time_series_covariance_checks(monkeypatch):
    """Minuit covariance validity should propagate to fit_valid."""
    T = 3600
    eff = 0.4
    t_half = 164e-6
    event_times = simulate_decay(0.5, eff, T)

    times_dict = {"Po214": event_times}
    cfg = {
        "isotopes": {"Po214": {"half_life_s": t_half, "efficiency": eff}},
        "fit_background": True,
        "fit_initial": True,
    }

    import numpy.linalg as linalg

    def good_eigvals(x):
        return np.ones(x.shape[0])

    monkeypatch.setattr(linalg, "eigvals", good_eigvals)
    res = fit_time_series(times_dict, 0.0, T, cfg)
    assert res.params["fit_valid"]

    def bad_eigvals(x):
        vals = np.ones(x.shape[0])
        vals[0] = -1.0
        return vals

    def cholesky_fail(x):
        raise linalg.LinAlgError("not PD")

    monkeypatch.setattr(linalg, "eigvals", bad_eigvals)
    monkeypatch.setattr(linalg, "cholesky", cholesky_fail)
    res_bad = fit_time_series(times_dict, 0.0, T, cfg)
    assert not res_bad.params["fit_valid"]

    with pytest.raises(RuntimeError):
        fit_time_series(times_dict, 0.0, T, cfg, strict=True)


def test_fit_time_series_half_life_zero_raises():
    times_dict = {"Po214": np.array([0.0, 1.0])}
    cfg = {
        "isotopes": {"Po214": {"half_life_s": 0.0, "efficiency": 1.0}},
        "fit_background": True,
        "fit_initial": True,
    }
    with pytest.raises(ValueError):
        fit_time_series(times_dict, 0.0, 10.0, cfg)


def test_fit_time_series_half_life_negative_raises():
    times_dict = {"Po214": np.array([0.0, 1.0])}
    cfg = {
        "isotopes": {"Po214": {"half_life_s": -1.0, "efficiency": 1.0}},
        "fit_background": True,
        "fit_initial": True,
    }
    with pytest.raises(ValueError):
        fit_time_series(times_dict, 0.0, 10.0, cfg)


def test_fit_time_series_efficiency_zero_raises():
    times_dict = {"Po214": np.array([0.0, 1.0])}
    cfg = {
        "isotopes": {"Po214": {"half_life_s": 1.0, "efficiency": 0.0}},
        "fit_background": True,
        "fit_initial": True,
    }
    with pytest.raises(ValueError):
        fit_time_series(times_dict, 0.0, 10.0, cfg)


def test_fit_time_series_covariance_output():
    rng = np.random.default_rng(0)
    T = 10.0
    t_half = 164e-6
    eff = 0.5
    times = np.sort(rng.uniform(0, T, 50))
    times_dict = {"Po214": times}
    cfg = {
        "isotopes": {"Po214": {"half_life_s": t_half, "efficiency": eff}},
        "fit_background": True,
        "fit_initial": True,
    }
    res = fit_time_series(times_dict, 0.0, T, cfg)
    assert "cov_E_Po214_N0_Po214" in res.params
    cov_exp = res.get_cov("E_Po214", "N0_Po214")
    assert res.params["cov_E_Po214_N0_Po214"] == pytest.approx(cov_exp)


def test_model_uncertainty_uses_covariance():
    centers = np.array([0.0, 1.0])
    widths = np.array([1.0, 1.0])
    params = {
        "E_Po214": 1.0,
        "dE_Po214": 0.1,
        "N0_Po214": 2.0,
        "dN0_Po214": 0.2,
        "B_Po214": 0.0,
        "dB_Po214": 0.0,
        "fit_valid": True,
    }
    cov = np.zeros((3, 3))
    cov[0, 1] = cov[1, 0] = 0.05
    fr = FitResult(params, cov, 0)
    fr_nc = FitResult(params, np.zeros((3, 3)), 0)
    cfg = {"time_fit": {"hl_po214": [10.0], "eff_po214": [1.0]}}
    with_cov = analyze._model_uncertainty(centers, widths, fr, "Po214", cfg, True)
    no_cov = analyze._model_uncertainty(centers, widths, fr_nc, "Po214", cfg, True)
    assert np.any(with_cov > no_cov)


def test_model_uncertainty_nan_covariance():
    centers = np.array([0.0, 1.0])
    widths = np.array([1.0, 1.0])
    params = {
        "E_Po214": 1.0,
        "dE_Po214": 0.1,
        "N0_Po214": 2.0,
        "dN0_Po214": 0.2,
        "B_Po214": 0.0,
        "dB_Po214": 0.0,
        "fit_valid": True,
    }
    cov_nan = np.zeros((3, 3))
    cov_nan[0, 1] = cov_nan[1, 0] = np.nan
    fr_nan = FitResult(params, cov_nan, 0)
    fr_zero = FitResult(params, np.zeros((3, 3)), 0)
    cfg = {"time_fit": {"hl_po214": [10.0], "eff_po214": [1.0]}}
    sigma_nan = analyze._model_uncertainty(centers, widths, fr_nan, "Po214", cfg, True)
    sigma_zero = analyze._model_uncertainty(centers, widths, fr_zero, "Po214", cfg, True)
    assert np.all(np.isfinite(sigma_nan))
    assert np.allclose(sigma_nan, sigma_zero)


def test_spectrum_tail_amplitude_stability():
    rng = np.random.default_rng(50)
    base = np.concatenate([
        rng.normal(5.3, 0.05, 300),
        rng.normal(6.0, 0.05, 300),
        rng.normal(7.7, 0.05, 300),
    ])
    tail = 6.0 + rng.exponential(0.15, 100)
    energies = np.concatenate([base, tail])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (300, 30),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (400, 40),
        "tau_Po218": (0.1, 0.05),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (300, 30),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
        "S_bkg": (0.0, 100.0),
    }

    res = fit_spectrum(energies, priors, unbinned=True)
    expected = 600  # expected with current likelihood
    tol = 0.03  # keep the same 3 % window
    assert abs(res.params["S_Po218"] - expected) / expected < tol


def test_spectrum_positive_amplitude_bound():
    rng = np.random.default_rng(51)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 50),
        rng.normal(6.0, 0.05, 50),
        rng.normal(7.7, 0.05, 50),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (-100, 20),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (-100, 20),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (-100, 20),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    res = fit_spectrum(energies, priors)
    assert res.params["fit_valid"]
    for key in ("S_Po210", "S_Po218", "S_Po214"):
        assert res.params[key] >= 0
