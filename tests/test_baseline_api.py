import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline
import baseline_utils
from radon import subtract_baseline_counts as rb_counts, subtract_baseline_rate as rb_rate
from radon.baseline import subtract_baseline_counts as rbc_counts, subtract_baseline_rate as rbc_rate
import radmon


def test_baseline_api_round_trip():
    counts = 10
    baseline_counts = 4
    efficiency = 0.5
    live_time = 20.0
    baseline_live_time = 10.0
    fit_rate = counts / live_time / efficiency
    fit_sigma = (counts**0.5) / (live_time * efficiency)

    res_utils = baseline_utils.subtract_baseline_counts(
        counts, efficiency, live_time, baseline_counts, baseline_live_time
    )
    res_radon = rbc_counts(
        counts, efficiency, live_time, baseline_counts, baseline_live_time
    )
    res_pkg = rb_counts(
        counts, efficiency, live_time, baseline_counts, baseline_live_time
    )
    assert res_utils == pytest.approx(res_radon)
    assert res_utils == pytest.approx(res_pkg)

    rate_utils = baseline_utils.subtract_baseline_rate(
        fit_rate,
        fit_sigma,
        counts,
        efficiency,
        live_time,
        baseline_counts,
        baseline_live_time,
    )
    rate_radon = rbc_rate(
        fit_rate,
        fit_sigma,
        counts,
        efficiency,
        live_time,
        baseline_counts,
        baseline_live_time,
    )
    rate_pkg = rb_rate(
        fit_rate,
        fit_sigma,
        counts,
        efficiency,
        live_time,
        baseline_counts,
        baseline_live_time,
    )
    assert rate_utils == pytest.approx(rate_radon)
    assert rate_utils == pytest.approx(rate_pkg)

    # alias exposures
    assert baseline.subtract_baseline is baseline_utils.subtract
    assert radmon.subtract_baseline is baseline.subtract_baseline
