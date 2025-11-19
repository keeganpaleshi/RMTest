
import warnings
import numpy as np
import yaml
from scipy.optimize import OptimizeWarning

from fitting import fit_spectrum
from src.rmtest.spectral.intensity import (
    build_spectral_intensity,
    integral_of_intensity,
)


def _minimal_priors():
    return {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (10, 1),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (10, 1),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (10, 1),
        "b0": (0.0, 0.1),
        "b1": (0.0, 0.1),
    }


_ISO_LIST = ("Po210", "Po218", "Po214")


def _fit_domain(energies: np.ndarray) -> tuple[float, float]:
    edges = np.histogram_bin_edges(energies, bins="fd")
    return float(edges[0]), float(edges[-1])


def _build_intensity_helpers(energies):
    domain = _fit_domain(energies)
    use_emg = {iso: False for iso in _ISO_LIST}
    spectral = build_spectral_intensity(_ISO_LIST, use_emg, domain)
    return spectral, domain, use_emg


def _physical_params_from_result(result):
    params = {
        "sigma0": float(result.params["sigma0"]),
        "b0": float(result.params["b0"]),
        "b1": float(result.params["b1"]),
    }
    for iso in _ISO_LIST:
        mu_key = f"mu_{iso}"
        s_key = f"S_{iso}"
        params[mu_key] = float(result.params[mu_key])
        params[s_key] = float(result.params[s_key])
        params[f"N_{iso}"] = float(result.params[s_key])
    return params


def _run_fit(unbinned: bool) -> str:
    priors = _minimal_priors()
    energies = np.concatenate([np.full(5, 5.3), np.full(5, 6.0), np.full(5, 7.7)])
    flags = {"fix_sigma0": True, "fix_F": True}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        result = fit_spectrum(energies, priors, flags=flags, unbinned=unbinned)
    return result.params.get("likelihood_path")


def test_default_and_unbinned_likelihood_path():
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    path = _run_fit(cfg["spectral_fit"].get("unbinned_likelihood", False))
    assert path == "binned_poisson"

    cfg2 = yaml.safe_load(open("config.yaml"))
    cfg2.setdefault("spectral_fit", {})["unbinned_likelihood"] = True
    path_unbinned = _run_fit(cfg2["spectral_fit"].get("unbinned_likelihood", False))
    assert path_unbinned in {"unbinned", "unbinned_extended"}


def test_unbinned_likelihood_modes_agree_numerically():
    priors = _minimal_priors()
    energies = np.concatenate([np.full(5, 5.3), np.full(5, 6.0), np.full(5, 7.7)])
    base_flags = {"fix_sigma0": True, "fix_F": True}

    result_default = fit_spectrum(energies, priors, flags=base_flags, unbinned=True)

    ext_flags = dict(base_flags)
    ext_flags["likelihood"] = "extended"
    result_extended = fit_spectrum(energies, priors, flags=ext_flags, unbinned=True)

    spectral, domain, use_emg = _build_intensity_helpers(energies)

    physical_default = _physical_params_from_result(result_default)
    lam_default = spectral(energies, physical_default, domain)
    manual_default = float(-np.sum(np.log(lam_default)))
    assert np.isclose(
        result_default.params["nll"],
        manual_default,
        rtol=1e-6,
        atol=1e-8,
    )

    physical_extended = _physical_params_from_result(result_extended)
    lam_extended = spectral(energies, physical_extended, domain)
    mu_total = integral_of_intensity(
        physical_extended,
        domain,
        iso_list=_ISO_LIST,
        use_emg=use_emg,
    )
    manual_extended = float(-(np.sum(np.log(lam_extended)) - mu_total))
    assert np.isclose(
        result_extended.params["nll"],
        manual_extended,
        rtol=1e-6,
        atol=1e-8,
    )
