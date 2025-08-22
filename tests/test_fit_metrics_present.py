import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting import fit_spectrum


def test_fit_metrics_present():
    rng = np.random.default_rng(0)
    energies = np.concatenate([
        rng.normal(5.3, 0.05, 40),
        rng.normal(6.0, 0.05, 40),
        rng.normal(7.7, 0.05, 40),
    ])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (40, 5),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (40, 5),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (40, 5),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }

    binned = fit_spectrum(energies, priors)
    assert np.isfinite(binned.params.get("aic"))
    assert np.isfinite(binned.params.get("chi2"))
    chi2_ndf = binned.params.get("chi2_ndf")
    if binned.ndf == 0:
        assert np.isnan(chi2_ndf)
    else:
        assert np.isfinite(chi2_ndf)

    unbinned = fit_spectrum(energies, priors, unbinned=True)
    assert np.isfinite(unbinned.params.get("aic"))
