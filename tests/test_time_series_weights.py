import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting import fit_time_series


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
    res_half = fit_time_series(
        {"Po214": times}, 0.0, 10, cfg, weights={"Po214": np.ones_like(times) * 0.5}
    )
    res_double = fit_time_series(
        {"Po214": times}, 0.0, 10, cfg, weights={"Po214": np.ones_like(times) * 2.0}
    )
    assert res_half.params["E_Po214"] == pytest.approx(res_double.params["E_Po214"], rel=1e-2)


def test_variable_weights_scale_independent():
    times = simulate_times(60, 20, seed=2)
    cfg = base_config(20)
    rng = np.random.default_rng(3)
    w = rng.uniform(0.2, 1.5, size=times.size)
    res_base = fit_time_series({"Po214": times}, 0.0, 20, cfg, weights={"Po214": w})
    res_scaled = fit_time_series({"Po214": times}, 0.0, 20, cfg, weights={"Po214": 3 * w})
    assert res_base.params["E_Po214"] == pytest.approx(res_scaled.params["E_Po214"], rel=1e-2)
