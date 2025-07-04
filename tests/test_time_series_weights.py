import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting import (
    fit_time_series,
    _neg_log_likelihood_time,
)


def simulate_times(n, T, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, T, n))


def base_config(T):
    return {
        "isotopes": {"Po214": {"half_life_s": 1.0, "efficiency": 1.0}},
        "fit_background": False,
        "fit_initial": False,
    }


def test_uniform_weight_scaling_invariant():
    times = simulate_times(50, 10, seed=1)
    cfg = base_config(10)
    base_w = np.ones_like(times)
    res0 = fit_time_series({"Po214": times}, 0.0, 10, cfg, weights={"Po214": base_w})
    res_half = fit_time_series(
        {"Po214": times}, 0.0, 10, cfg, weights={"Po214": 0.5 * base_w}
    )
    res_double = fit_time_series(
        {"Po214": times}, 0.0, 10, cfg, weights={"Po214": 2.0 * base_w}
    )
    assert res0.params["E_Po214"] == pytest.approx(res_half.params["E_Po214"], rel=1e-2)
    assert res0.params["E_Po214"] == pytest.approx(res_double.params["E_Po214"], rel=1e-2)


def test_variable_weights_scale_independent():
    times = simulate_times(60, 20, seed=2)
    cfg = base_config(20)
    rng = np.random.default_rng(3)
    w = rng.uniform(0.2, 1.5, size=times.size)
    res_base = fit_time_series({"Po214": times}, 0.0, 20, cfg, weights={"Po214": w})
    res_scaled = fit_time_series({"Po214": times}, 0.0, 20, cfg, weights={"Po214": 3 * w})
    assert res_base.params["E_Po214"] == pytest.approx(res_scaled.params["E_Po214"], rel=1e-2)


def test_weighted_nll_matches_numeric_integration():
    """Analytic log-likelihood should match numerical integration for weights."""
    E = 0.3
    B = 0.05
    N0 = 0.1
    lam = np.log(2.0) / 1.0
    eff = 0.9
    t_start = 0.0
    t_end = 4.0
    times = np.array([1.0, 3.0])
    weights = np.array([0.5, 2.0])

    times_dict = {"Po214": times}
    weights_dict = {"Po214": weights}

    params = (E, B, N0)
    iso_list = ["Po214"]
    lam_map = {"Po214": lam}
    eff_map = {"Po214": eff}
    fix_b_map = {"Po214": False}
    fix_n0_map = {"Po214": False}
    param_indices = {"E_Po214": 0, "B_Po214": 1, "N0_Po214": 2}

    analytic_nll = _neg_log_likelihood_time(
        params,
        times_dict,
        weights_dict,
        t_start,
        t_end,
        iso_list,
        lam_map,
        eff_map,
        fix_b_map,
        fix_n0_map,
        param_indices,
    )

    def rate(t):
        return eff * (E * (1 - np.exp(-lam * t)) + lam * N0 * np.exp(-lam * t)) + B

    weight_mean = np.mean(weights)
    integral_num = quad(rate, 0.0, t_end - t_start)[0] * weight_mean
    rate_vals = rate(times - t_start)
    numeric_nll = integral_num - np.sum(weights * np.log(rate_vals))

    assert analytic_nll == pytest.approx(numeric_nll, rel=1e-6)


def test_weights_none_equivalent_to_ones():
    times = simulate_times(40, 15, seed=4)
    cfg = base_config(15)
    res_none = fit_time_series({"Po214": times}, 0.0, 15, cfg, weights=None)
    res_one = fit_time_series(
        {"Po214": times}, 0.0, 15, cfg, weights={"Po214": np.ones_like(times)}
    )
    assert res_none.params["E_Po214"] == pytest.approx(
        res_one.params["E_Po214"], rel=1e-6
    )


def test_corrected_sigma_weighting():
    """Weighted fit should reproduce variance from baseline subtraction."""
    T = 100.0
    n = 100
    baseline_counts = 25
    baseline_live_time = 100.0
    seed = 1

    times = simulate_times(n, T, seed=seed)
    cfg = base_config(T)

    # Standard unweighted fit
    res_std = fit_time_series({"Po214": times}, 0.0, T, cfg, weights={"Po214": np.ones_like(times)})

    # Expected uncertainty from baseline subtraction
    from baseline_utils import subtract_baseline_counts

    _, corrected_sigma = subtract_baseline_counts(
        n,
        1.0,
        T,
        baseline_counts,
        baseline_live_time,
    )

    # Uniform weight factor that reproduces the expected variance
    w = n * baseline_live_time**2 / (n * baseline_live_time**2 + baseline_counts * T**2)

    res_weight = fit_time_series({"Po214": times}, 0.0, T, cfg, weights={"Po214": np.full_like(times, w)})

    assert res_std.params["E_Po214"] == pytest.approx(
        res_weight.params["E_Po214"], rel=1e-3
    )
    assert res_weight.params["dE_Po214"] == pytest.approx(corrected_sigma, rel=2e-2)
