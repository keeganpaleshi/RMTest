import numpy as np

from fitting import fit_spectrum


def _priors(S=5.0):
    return {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (S, S),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (S, S),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (S, S),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }


def test_fit_metrics_present():
    rng = np.random.default_rng(0)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 5),
        rng.normal(6.0, 0.05, 5),
        rng.normal(7.7, 0.05, 5),
    ])
    priors = _priors()

    res_binned = fit_spectrum(energies, priors, bins=30)
    assert "aic" in res_binned.params
    assert np.isfinite(res_binned.params["aic"])
    assert "chi2" in res_binned.params
    assert "chi2_ndf" in res_binned.params
    assert np.isfinite(res_binned.params["chi2"])
    chi2_ndf = res_binned.params["chi2_ndf"]
    if res_binned.ndf == 0:
        assert np.isnan(chi2_ndf)
    else:
        assert np.isfinite(chi2_ndf)

    res_unbinned = fit_spectrum(energies, priors, unbinned=True)
    assert "aic" in res_unbinned.params
    assert np.isfinite(res_unbinned.params["aic"])
