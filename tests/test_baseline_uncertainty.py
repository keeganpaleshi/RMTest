import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radon.baseline import subtract_baseline


def test_baseline_uncertainty():
    counts = 100
    baseline_counts = 25
    efficiency = 0.5
    live_time = 3600.0
    baseline_live_time = 7200.0

    corrected_rate, corrected_sigma = subtract_baseline(
        counts=counts,
        baseline_counts=baseline_counts,
        efficiency=efficiency,
        live_time=live_time,
        baseline_live_time=baseline_live_time,
    )

    expected_rate = counts / (live_time * efficiency) - baseline_counts / (
        baseline_live_time * efficiency
    )
    sigma_rate = np.sqrt(counts) / (live_time * efficiency)
    sigma_baseline = np.sqrt(baseline_counts) / (baseline_live_time * efficiency)
    expected_sigma = np.hypot(sigma_rate, sigma_baseline)

    assert corrected_rate == pytest.approx(expected_rate)
    assert corrected_sigma == pytest.approx(expected_sigma)
