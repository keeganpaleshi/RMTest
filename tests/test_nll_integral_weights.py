import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fitting import _neg_log_likelihood_time


def _numeric_integral(E, N0, B, lam, eff, T, steps=10000):
    ts = np.linspace(0.0, T, steps)
    rate = eff * (E * (1.0 - np.exp(-lam * ts)) + lam * N0 * np.exp(-lam * ts)) + B
    return np.trapz(rate, ts)


def test_nll_analytic_vs_numeric():
    iso = "Po214"
    t_start = 0.0
    t_end = 3.0
    times = np.array([1.0, 2.0])
    weights = np.array([2.0, 3.0])

    params = (1.0,)  # E_iso only
    lam = 0.2
    eff = 1.0
    N0 = 0.0
    B = 0.0

    times_dict = {iso: times}
    weights_dict = {iso: weights}
    iso_list = [iso]
    lam_map = {iso: lam}
    eff_map = {iso: eff}
    fix_b_map = {iso: True}
    fix_n0_map = {iso: True}
    param_indices = {"E_" + iso: 0}

    # Analytic NLL using internal function
    nll_analytic = _neg_log_likelihood_time(
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

    # Numeric NLL using trapezoidal integration
    T_rel = t_end - t_start
    integral_num = _numeric_integral(params[0], N0, B, lam, eff, T_rel)
    weight_sum = np.sum(weights)
    rate_events = eff * (
        params[0] * (1.0 - np.exp(-lam * (times - t_start)))
        + lam * N0 * np.exp(-lam * (times - t_start))
    ) + B
    nll_numeric = integral_num * weight_sum - np.sum(weights * np.log(rate_events))

    assert nll_analytic == pytest.approx(nll_numeric, rel=1e-6)

