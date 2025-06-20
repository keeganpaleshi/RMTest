import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
from radon.baseline import subtract_baseline


def test_subtract_baseline_uncertainty():
    counts = 50
    baseline_counts = 20
    efficiency = 0.5
    live_time = 100.0
    baseline_live_time = 50.0

    expected_rate = counts / live_time / efficiency - baseline_counts / baseline_live_time / efficiency
    expected_sigma_sq = (
        counts / live_time**2 / efficiency**2
        + baseline_counts / baseline_live_time**2 / efficiency**2
    )
    expected_sigma = np.sqrt(expected_sigma_sq)

    rate, sigma = subtract_baseline(
        counts, efficiency, live_time, baseline_counts, baseline_live_time
    )

    assert rate == pytest.approx(expected_rate)
    assert sigma == pytest.approx(expected_sigma)

