import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor

from emg_stable import StableEMG, parallel_emg_pdf_map


def test_parallel_emg_pdf_map_matches_serial():
    x = np.linspace(-0.5, 0.5, 20)
    params = [
        {"mu": 0.0, "sigma": 0.05, "tau": 0.02, "amplitude": 2.0},
        (0.1, 0.03, 0.01, 1.5),
        [0.2, 0.04, 0.015],
    ]

    results = parallel_emg_pdf_map(x, params, prefer="threads")

    emg = StableEMG()
    expected = [
        emg.pdf(x, 0.0, 0.05, 0.02, 2.0),
        emg.pdf(x, 0.1, 0.03, 0.01, 1.5),
        emg.pdf(x, 0.2, 0.04, 0.015, 1.0),
    ]

    assert len(results) == len(expected)
    for res, exp in zip(results, expected):
        assert np.allclose(res, exp)


def test_parallel_emg_pdf_map_empty_params():
    x = np.linspace(-0.5, 0.5, 10)
    assert parallel_emg_pdf_map(x, []) == []


def test_parallel_emg_pdf_map_with_executor():
    x = np.linspace(-0.5, 0.5, 10)
    params = [{"mu": 0.0, "sigma": 0.05, "tau": 0.02}]

    with ThreadPoolExecutor(max_workers=1) as pool:
        results = parallel_emg_pdf_map(x, params, executor=pool)

    assert len(results) == 1
    assert np.allclose(results[0], StableEMG().pdf(x, 0.0, 0.05, 0.02, 1.0))


def test_parallel_emg_pdf_map_invalid_prefer():
    x = np.linspace(-0.5, 0.5, 10)
    params = [{"mu": 0.0, "sigma": 0.05, "tau": 0.02}]

    with pytest.raises(ValueError):
        parallel_emg_pdf_map(x, params, prefer="invalid")


def test_parallel_emg_pdf_map_missing_key():
    x = np.linspace(-0.5, 0.5, 10)
    params = [{"sigma": 0.05, "tau": 0.02}]

    with pytest.raises(KeyError):
        parallel_emg_pdf_map(x, params)
