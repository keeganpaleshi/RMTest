import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
from radon.baseline import subtract_baseline_counts


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

    rate, sigma = subtract_baseline_counts(
        counts, efficiency, live_time, baseline_counts, baseline_live_time
    )

    assert rate == pytest.approx(expected_rate)
    assert sigma == pytest.approx(expected_sigma)


def test_zero_live_time_behaviour():
    counts = 10
    baseline_counts = 5
    efficiency = 1.0
    # live_time zero -> should raise descriptive ValueError
    with pytest.raises(ValueError):
        subtract_baseline_counts(
            counts, efficiency, 0.0, baseline_counts, 50.0
        )


def test_zero_baseline_live_time_behaviour():
    counts = 10
    baseline_counts = 0
    efficiency = 1.0
    # baseline_live_time zero -> should raise descriptive ValueError
    with pytest.raises(ValueError):
        subtract_baseline_counts(
            counts, efficiency, 100.0, baseline_counts, 0.0
        )


def test_zero_efficiency_behaviour():
    counts = 10
    baseline_counts = 2
    efficiency = 0.0
    with pytest.raises(ValueError):
        subtract_baseline_counts(
            counts, efficiency, 100.0, baseline_counts, 50.0
        )


def test_negative_efficiency_behaviour():
    counts = 10
    baseline_counts = 2
    efficiency = -0.1
    with pytest.raises(ValueError):
        subtract_baseline_counts(
            counts, efficiency, 100.0, baseline_counts, 50.0
        )

