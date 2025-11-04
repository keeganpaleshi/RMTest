"""Spectral utilities for radon decay spectroscopy.

This package provides tools for spectral fitting of radon decay chains,
including properly normalized shape functions and unbinned likelihood methods.
"""

from .shapes import emg_pdf_E, gaussian_pdf_E
from .intensity import (
    build_spectral_intensity,
    spectral_intensity_E,
    integral_of_intensity,
)
from .nll_unbinned import nll_extended_unbinned, nll_extended_unbinned_simple

__all__ = [
    "emg_pdf_E",
    "gaussian_pdf_E",
    "build_spectral_intensity",
    "spectral_intensity_E",
    "integral_of_intensity",
    "nll_extended_unbinned",
    "nll_extended_unbinned_simple",
]
