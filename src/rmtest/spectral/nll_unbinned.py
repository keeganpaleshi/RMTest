"""Extended unbinned negative log-likelihood for spectral fitting.

This module implements the proper extended unbinned likelihood for the radon
spectrum, ensuring correct normalization and scale for all parameters.
"""

import numpy as np
from .intensity import spectral_intensity_E, integral_of_intensity


def nll_extended_unbinned(E, params, domain, use_emg=None):
    """Extended unbinned negative log-likelihood.

    The extended unbinned likelihood for N observed events at energies E_i is:

        L = (μ^N / N!) * exp(-μ) * Π_i λ(E_i)

    where:
        μ = ∫ λ(E) dE  (total expected counts)
        λ(E) is the rate density [counts/MeV]

    The negative log-likelihood (dropping constant terms) is:

        -ln L = μ - Σ_i ln λ(E_i)

    Parameters
    ----------
    E : array-like
        Observed energy values in MeV.
    params : dict or array-like
        Model parameters (see :func:`spectral_intensity_E` for format).
    domain : tuple
        (Emin, Emax) defining the fit range in MeV.
    use_emg : dict, optional
        Mapping {iso: bool} for EMG vs Gaussian choice.

    Returns
    -------
    float
        Negative log-likelihood value. Returns np.inf if the model is invalid
        (negative or non-finite intensities).

    Notes
    -----
    - The intensity λ(E) must be positive and finite for all observed events.
    - The integral μ must be positive and finite.
    - This implementation does NOT include bin-width factors—those are only
      relevant for binned Poisson likelihoods.
    """
    E = np.asarray(E, dtype=float)

    # Evaluate intensity at observed energies
    lam = spectral_intensity_E(E, params, domain, use_emg=use_emg)

    # Check validity
    if np.any(lam <= 0) or not np.isfinite(lam).all():
        return np.inf

    # Compute the total expected counts
    mu_tot = integral_of_intensity(params, domain, use_emg=use_emg)

    if not np.isfinite(mu_tot) or mu_tot <= 0:
        return np.inf

    # Extended unbinned NLL: μ - Σ ln(λ(E_i))
    nll = mu_tot - np.sum(np.log(lam))

    return float(nll)
