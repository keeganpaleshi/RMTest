"""Wrappers around the legacy stable EMG implementation."""

from emg_stable import StableEMG, emg_left_stable, parallel_emg_pdf_map

__all__ = [
    "StableEMG",
    "emg_left_stable",
    "parallel_emg_pdf_map",
]
