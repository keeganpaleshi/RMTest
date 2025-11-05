import numpy as np
from rmtest.spectral.window_norm import normalize_pdf_to_window
from rmtest.spectral.shapes import emg_cdf_E
from rmtest.spectral.intensity import build_spectral_intensity, integral_of_intensity

def test_emg_window_mass_is_near_one_for_this_case():
    mu, sigma, tau = 6.0, 0.05, 0.10
    E_lo, E_hi = 4.8, 8.3
    pwin = float(emg_cdf_E(E_hi, mu, sigma, tau) - emg_cdf_E(E_lo, mu, sigma, tau))
    assert 0.99999 < pwin < 1.0

def test_extended_mu_equals_sum_of_yields_after_window_renorm():
    domain = (4.8, 8.3)
    iso_list = ("Po210", "Po218", "Po214")
    spectral = build_spectral_intensity(iso_list, use_emg=True, domain=domain)

    params = dict(
        N_Po210=300.0, mu_Po210=5.3,
        N_Po218=400.0, mu_Po218=6.0, tau_Po218=0.10,
        N_Po214=300.0, mu_Po214=7.7,
        sigma0=0.05, b0=0.0, b1=0.0
    )
    E = np.linspace(domain[0], domain[1], 6001)
    lam = spectral(E, params, domain)
    mu = integral_of_intensity(params, domain, iso_list=iso_list, use_emg=True)
    num_mu = np.trapz(lam, E)
    assert abs(mu - (300+400+300)) < 1e-6
    assert abs(num_mu - mu) / mu < 1e-3
