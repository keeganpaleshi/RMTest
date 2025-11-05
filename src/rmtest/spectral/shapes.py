# src/rmtest/spectral/shapes.py
import numpy as np
from scipy.stats import exponnorm, norm

def emg_pdf_E(E, mu, sigma, tau):
    """
    Exponentially modified Gaussian PDF.
    E, mu in MeV; sigma,tau in MeV. Returns unit-integral pdf over (-inf,inf).
    SciPy parametrization: K = tau/sigma, loc = mu, scale = sigma.
    """
    E = np.asarray(E, dtype=float)
    if sigma <= 0 or tau <= 0:
        return np.zeros_like(E, dtype=float)
    K = tau / sigma
    return exponnorm.pdf(E, K, loc=mu, scale=sigma)

def emg_cdf_E(E, mu, sigma, tau):
    E = np.asarray(E, dtype=float)
    if sigma <= 0 or tau <= 0:
        return np.zeros_like(E, dtype=float)
    K = tau / sigma
    return exponnorm.cdf(E, K, loc=mu, scale=sigma)

def gaussian_pdf_E(E, mu, sigma):
    E = np.asarray(E, dtype=float)
    if sigma <= 0:
        return np.zeros_like(E, dtype=float)
    return norm.pdf(E, loc=mu, scale=sigma)

def gaussian_cdf_E(E, mu, sigma):
    E = np.asarray(E, dtype=float)
    if sigma <= 0:
        return np.zeros_like(E, dtype=float)
    return norm.cdf(E, loc=mu, scale=sigma)
