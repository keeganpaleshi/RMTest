import math
import timeit
import numpy as np
from calibration import gaussian, emg_left
from fitting import fit_spectrum
from iminuit import Minuit


def fit_spectrum_slow(energies, priors):
    """Pre-vectorisation implementation using Python loops."""
    e = np.asarray(energies, dtype=float)
    edges = np.histogram_bin_edges(e, bins="fd")

    def p(name, default):
        return priors.get(name, (default, 1.0))

    use_emg = {
        "Po210": "tau_Po210" in priors,
        "Po218": "tau_Po218" in priors,
        "Po214": "tau_Po214" in priors,
    }

    param_order = ["sigma0", "F"]
    for iso in ("Po210", "Po218", "Po214"):
        param_order.extend([f"mu_{iso}", f"S_{iso}"])
        if use_emg[iso]:
            param_order.append(f"tau_{iso}")
    param_order.extend(["b0", "b1"])

    p0 = [p(name, 1.0)[0] for name in param_order]
    iso_list = ["Po210", "Po218", "Po214"]

    def _model_density(x, *params):
        idx = 0
        sigma0 = params[idx]; idx += 1
        F_val = params[idx]; idx += 1
        y = np.zeros_like(x)
        for iso in iso_list:
            mu = params[idx]; idx += 1
            S = params[idx]; idx += 1
            if use_emg[iso]:
                tau = params[idx]; idx += 1
                sigma = np.array([math.sqrt(sigma0 ** 2 + F_val * xi) for xi in x])
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    y_emg = emg_left(x, mu, sigma, tau)
                y_emg = np.nan_to_num(y_emg, nan=0.0, posinf=0.0, neginf=0.0)
                y += S * y_emg
            else:
                sigma = np.array([math.sqrt(sigma0 ** 2 + F_val * xi) for xi in x])
                y += S * gaussian(x, mu, sigma)
        b0 = params[idx]
        b1 = params[idx + 1]
        return y + b0 + b1 * x

    def _nll(*params):
        rate = _model_density(e, *params)
        if np.any(rate <= 0) or not np.isfinite(rate).all():
            return 1e50
        idx = 2
        S_sum = 0.0
        for iso in iso_list:
            idx += 1  # mu
            S_sum += params[idx]
            idx += 1
            if use_emg[iso]:
                idx += 1
        b0 = params[idx]
        b1 = params[idx + 1]
        E_lo = edges[0]
        E_hi = edges[-1]
        bkg_int = b0 * (E_hi - E_lo) + 0.5 * b1 * (E_hi**2 - E_lo**2)
        expected = S_sum + bkg_int
        return expected - np.sum(np.log(rate))

    m = Minuit(_nll, *p0, name=param_order)
    m.limits["F"] = (0, None)
    m.limits["sigma0"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()
    return m.fval


def main():
    rng = np.random.default_rng(0)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 300_000),
        rng.normal(6.0, 0.05, 300_000),
        rng.normal(7.7, 0.05, 300_000),
        rng.uniform(5.0, 8.0, 100_000),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (300000, 3000),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (300000, 3000),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (300000, 3000),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    t_slow = timeit.timeit(lambda: fit_spectrum_slow(energies, priors), number=1)
    t_fast = timeit.timeit(lambda: fit_spectrum(energies, priors, unbinned=True), number=1)

    print(f"Slow runtime: {t_slow:.2f}s")
    print(f"Fast runtime: {t_fast:.2f}s")
    speedup = t_slow / t_fast if t_fast > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x")
    assert speedup >= 10.0, "Vectorised version is not at least 10x faster"


if __name__ == "__main__":
    main()
