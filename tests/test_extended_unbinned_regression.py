"""Regression coverage for the extended unbinned likelihood path."""

from __future__ import annotations

import numpy as np
import pytest

from fitting import fit_spectrum


def _sample_linear_background(
    rng: np.random.Generator,
    size: int,
    b0: float,
    b1: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Draw samples from a linear density ``b0 + b1 * E`` on ``[lo, hi]``."""

    if size <= 0:
        return np.empty(0, dtype=float)

    if abs(b1) < 1e-12:
        return rng.uniform(lo, hi, size)

    total = b0 * (hi - lo) + 0.5 * b1 * (hi**2 - lo**2)
    u = rng.random(size)
    A = 0.5 * b1
    B = b0
    C0 = -b0 * lo - 0.5 * b1 * lo**2
    C = C0 - u * total
    disc = np.maximum(B**2 - 4 * A * C, 0.0)
    roots = (-B + np.sqrt(disc)) / (2 * A)
    return np.clip(roots, lo, hi)


def _sample_peak(
    rng: np.random.Generator,
    size: int,
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Sample ``size`` events from a Gaussian truncated to ``[lo, hi]``."""

    if size <= 0:
        return np.empty(0, dtype=float)

    out: list[np.ndarray] = []
    remaining = size
    while remaining > 0:
        draws = rng.normal(mu, sigma, remaining * 2)
        draws = draws[(draws >= lo) & (draws <= hi)]
        if draws.size == 0:
            continue
        take = min(draws.size, remaining)
        out.append(draws[:take])
        remaining -= take
    return np.concatenate(out)


def test_unbinned_extended_recovers_peak_and_background_counts():
    rng = np.random.default_rng(42)

    E_lo, E_hi = 4.5, 8.0
    sigma0_true = 0.05
    peaks = {
        "Po210": {"mu": 5.3, "area": 420},
        "Po214": {"mu": 6.0, "area": 310},
    }
    b0_true = 8.0
    b1_true = 0.6
    S_bkg_true = 480.0

    events = []
    generated_counts: dict[str, int] = {}
    for iso, spec in peaks.items():
        count = rng.poisson(spec["area"])
        generated_counts[iso] = int(count)
        events.append(
            _sample_peak(rng, count, spec["mu"], sigma0_true, E_lo, E_hi)
        )

    n_bkg = rng.poisson(S_bkg_true)
    generated_counts["S_bkg"] = int(n_bkg)
    events.append(
        _sample_linear_background(rng, n_bkg, b0_true, b1_true, E_lo, E_hi)
    )

    energies = np.concatenate([ev for ev in events if ev.size])
    rng.shuffle(energies)

    priors = {
        "sigma0": (sigma0_true, 0.0),
        "F": (0.0, 0.0),
        "b0": (b0_true, 1.0),
        "b1": (b1_true, 0.5),
        "S_bkg": (S_bkg_true, max(20.0, np.sqrt(S_bkg_true))),
    }
    for iso, spec in peaks.items():
        priors[f"mu_{iso}"] = (spec["mu"], 0.02)
        priors[f"S_{iso}"] = (
            spec["area"],
            max(10.0, np.sqrt(spec["area"]))
        )

    flags = {
        "likelihood": "extended",
        "fix_sigma0": True,
        "fix_F": True,
    }

    bin_edges = np.linspace(E_lo, E_hi, 161)
    result = fit_spectrum(
        energies,
        priors,
        unbinned=True,
        flags=flags,
        bin_edges=bin_edges,
    )

    assert result.likelihood == "unbinned"

    for iso in peaks:
        fitted = result.params[f"S_{iso}"]
        observed = generated_counts[iso]
        tol = 5.0 * np.sqrt(max(observed, 1))
        assert fitted == pytest.approx(observed, abs=tol)

    fitted_bkg = result.params["S_bkg"]
    observed_bkg = generated_counts["S_bkg"]
    tol_bkg = 5.0 * np.sqrt(max(observed_bkg, 1))
    assert fitted_bkg == pytest.approx(observed_bkg, abs=tol_bkg)
