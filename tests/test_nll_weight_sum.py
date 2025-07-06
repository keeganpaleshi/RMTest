import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting import _neg_log_likelihood_time


def test_weighted_nll_analytic_matches_numeric():
    E = 0.4
    B = 0.1
    N0 = 0.2
    half_life = 2.0
    lam = np.log(2.0) / half_life
    eff = 0.8

    t_start = 0.0
    t_end = 5.0
    times = np.array([1.0, 3.0])
    weights = np.array([1.5, 0.8])

    params = (E, B, N0)
    times_dict = {"Po214": times}
    weights_dict = {"Po214": weights}
    iso_list = ["Po214"]
    lam_map = {"Po214": lam}
    eff_map = {"Po214": eff}
    fix_b_map = {"Po214": False}
    fix_n0_map = {"Po214": False}
    param_indices = {"E_Po214": 0, "B_Po214": 1, "N0_Po214": 2}
    var_eff_map = {"Po214": False}

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
        var_eff_map,
    )

    def rate(t):
        return eff * (E * (1 - np.exp(-lam * t)) + lam * N0 * np.exp(-lam * t)) + B

    weight_mean = np.mean(weights)
    integral_num = quad(rate, 0.0, t_end - t_start)[0] * weight_mean
    rate_vals = rate(times - t_start)
    numeric_nll = integral_num - np.sum(weights * np.log(rate_vals))

    assert analytic_nll == pytest.approx(numeric_nll, rel=1e-6)
