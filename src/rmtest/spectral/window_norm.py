# src/rmtest/spectral/window_norm.py
"""
Window normalization utilities for deterministic peak PDF normalization.

This module provides functions to normalize peak PDFs to a specific energy window
using CDF-based probability mass calculation, ensuring deterministic integration
without per-iteration quadrature.
"""
import numpy as np
from .shapes import emg_pdf_E, emg_cdf_E, gaussian_pdf_E, gaussian_cdf_E

def _component_p_in_window(kind, E_lo, E_hi, mu, sigma, tau=None):
    """Probability mass of a single component inside [E_lo, E_hi]."""
    if kind == "emg":
        p = float(emg_cdf_E(E_hi, mu, sigma, tau) - emg_cdf_E(E_lo, mu, sigma, tau))
    elif kind == "gauss":
        p = float(gaussian_cdf_E(E_hi, mu, sigma) - gaussian_cdf_E(E_lo, mu, sigma))
    else:
        raise ValueError(f"unknown kind {kind}")
    return max(p, 1e-12)

def component_pdf(kind, E, mu, sigma, tau=None):
    """Raw (global) pdf on E."""
    if kind == "emg":
        return emg_pdf_E(E, mu, sigma, tau)
    elif kind == "gauss":
        return gaussian_pdf_E(E, mu, sigma)
    else:
        raise ValueError(f"unknown kind {kind}")

def normalize_pdf_to_window(kind, mu, sigma, E_lo, E_hi, tau=None):
    """
    Return a function pdf_win(E) that integrates to 1 over [E_lo,E_hi].
    """
    pwin = _component_p_in_window(kind, E_lo, E_hi, mu, sigma, tau)
    def pdf_win(E):
        return component_pdf(kind, E, mu, sigma, tau) / pwin
    return pdf_win, pwin
