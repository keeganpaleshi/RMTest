"""Numerical helpers for EMG calibration exposed via :mod:`rmtest`."""

from __future__ import annotations

import calibration as _calibration

from calibration import gaussian

__all__ = ["emg_left", "gaussian"]


def emg_left(x, mu, sigma, tau):
    """Delegate to :func:`calibration.emg_left` while honouring legacy globals."""

    return _calibration.emg_left(x, mu, sigma, tau)
