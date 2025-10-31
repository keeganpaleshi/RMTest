import numpy as np
import pytest

import emg_stable


def _expected_emg(x, mu, sigma, tau, amplitude):
    return emg_stable.emg_left_stable(
        x,
        mu,
        sigma,
        tau,
        amplitude=amplitude,
        use_log_scale=False,
    )


def test_parallel_emg_map_matches_serial_results():
    x = np.linspace(0.0, 2.0, 64)
    params = [
        {"mu": 0.5, "sigma": 0.1, "tau": 0.05, "amplitude": 1.2},
        (0.9, 0.08, 0.03, 0.7),
        (1.2, 0.12, 0.02),
    ]

    parallel = emg_stable.parallel_emg_map(
        x,
        params,
        amplitude=1.0,
        max_workers=2,
        chunk_size=1,
    )

    expected = np.vstack([
        _expected_emg(x, 0.5, 0.1, 0.05, 1.2),
        _expected_emg(x, 0.9, 0.08, 0.03, 0.7),
        _expected_emg(x, 1.2, 0.12, 0.02, 1.0),
    ])

    assert parallel.shape == expected.shape
    assert np.allclose(parallel, expected)


def test_parallel_emg_map_handles_empty_input():
    x = np.linspace(-1.0, 1.0, 10)
    result = emg_stable.parallel_emg_map(x, [], max_workers=2)
    assert result.shape == (0, x.size)


def test_parallel_emg_map_invalid_sequence_length():
    x = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        emg_stable.parallel_emg_map(x, [(0.1, 0.05)], max_workers=1)
