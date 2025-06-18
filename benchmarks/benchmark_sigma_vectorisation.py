import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import timeit
import numpy as np
from fitting import fit_spectrum


def fit_spectrum_slow(energies, priors, bins=300):
    e = np.asarray(energies, dtype=float)
    edges = np.histogram_bin_edges(e, bins=bins)
    hist, _ = np.histogram(e, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)

    sigma0 = priors["sigma0"][0]
    F = priors["F"][0]
    mu = priors["mu_Po214"][0]
    S = priors["S_Po214"][0]
    b0 = priors["b0"][0]
    b1 = priors["b1"][0]

    def _model(x, sigma0, F, mu, S, b0, b1):
        y = []
        for xi in x:
            sigma = math.sqrt(sigma0 ** 2 + F * xi)
            g = math.exp(-0.5 * ((xi - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
            y.append(S * g + b0 + b1 * xi)
        return np.array(y) * width

    for _ in range(1000):
        _model(centers, sigma0, F, mu, S, b0, b1)


def main():
    rng = np.random.default_rng(0)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, int(3.3e5)),
        rng.normal(6.0, 0.05, int(3.3e5)),
        rng.normal(7.7, 0.05, int(4e5)),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (4e5, 1e4),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    fast = timeit.timeit(lambda: fit_spectrum(energies, priors, bins=300), number=1)
    slow = timeit.timeit(lambda: fit_spectrum_slow(energies, priors, bins=300), number=1)

    print(f"Optimised fit: {fast:.3f} s")
    print(f"Unvectorised fit: {slow:.3f} s")
    speedup = slow / fast if fast > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x")
    assert speedup >= 10.0, "Sigma vectorisation is less than 10x faster"


if __name__ == "__main__":
    main()
