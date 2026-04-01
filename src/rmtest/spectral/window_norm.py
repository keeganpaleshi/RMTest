import numpy as np
from .shapes import (
    emg_pdf_E, emg_cdf_E,
    right_emg_pdf_E, right_emg_cdf_E,
    gaussian_pdf_E, gaussian_cdf_E,
    split_gaussian_pdf_E, split_gaussian_cdf_E,
    shelf_pdf_E,
)

WINDOW_NORM_POINTS = 4096  # power-of-twoish; deterministic across runs


def _component_p_in_window(kind, E_lo, E_hi, mu, sigma, tau=None, **kwargs):
    """Probability mass of a single component inside [E_lo, E_hi]."""
    if kind == "emg":
        p = float(emg_cdf_E(E_hi, mu, sigma, tau) - emg_cdf_E(E_lo, mu, sigma, tau))
    elif kind == "right_emg":
        p = float(right_emg_cdf_E(E_hi, mu, sigma, tau) - right_emg_cdf_E(E_lo, mu, sigma, tau))
    elif kind == "gauss":
        p = float(gaussian_cdf_E(E_hi, mu, sigma) - gaussian_cdf_E(E_lo, mu, sigma))
    elif kind == "split_gauss":
        sigma_right = kwargs.get("sigma_right", sigma)
        p = float(split_gaussian_cdf_E(E_hi, mu, sigma, sigma_right) - split_gaussian_cdf_E(E_lo, mu, sigma, sigma_right))
    elif kind == "shelf":
        # shelf_pdf_E is already normalized to [E_lo, E_hi], so p_in_window = 1.
        p = 1.0
    else:
        raise ValueError(f"unknown kind {kind}")
    return max(p, 1e-12)


def component_pdf(kind, E, mu, sigma, tau=None, E_lo=None, E_hi=None, **kwargs):
    """Raw (global) pdf on E."""
    if kind == "emg":
        return emg_pdf_E(E, mu, sigma, tau)
    elif kind == "right_emg":
        return right_emg_pdf_E(E, mu, sigma, tau)
    elif kind == "gauss":
        return gaussian_pdf_E(E, mu, sigma)
    elif kind == "split_gauss":
        sigma_right = kwargs.get("sigma_right", sigma)
        return split_gaussian_pdf_E(E, mu, sigma, sigma_right)
    elif kind == "shelf":
        if E_lo is None or E_hi is None:
            raise ValueError("shelf kind requires E_lo and E_hi")
        shelf_range = kwargs.get("shelf_range")
        shelf_cutoff_delta = kwargs.get("shelf_cutoff_delta")
        return shelf_pdf_E(E, mu, sigma, E_lo, E_hi, shelf_range=shelf_range,
                           shelf_cutoff_delta=shelf_cutoff_delta)
    else:
        raise ValueError(f"unknown kind {kind}")


def normalize_pdf_to_window(kind, mu, sigma, E_lo, E_hi, tau=None, **kwargs):
    """
    Return a function pdf_win(E) that integrates to 1 over [E_lo,E_hi].
    """
    pwin = _component_p_in_window(kind, E_lo, E_hi, mu, sigma, tau, **kwargs)

    _kwargs = dict(kwargs)

    def pdf_win(E):
        return component_pdf(kind, E, mu, sigma, tau, E_lo=E_lo, E_hi=E_hi, **_kwargs) / pwin

    return pdf_win, pwin
