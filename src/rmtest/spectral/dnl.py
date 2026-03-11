"""ADC Differential Non-Linearity (DNL) correction.

DNL arises from non-uniform effective channel widths in the ADC hardware,
causing systematic count excesses or deficits at specific ADC channels.

The correction factor for each bin represents the relative effective width:

- ``dnl > 1.0``: channel is wider than nominal → more counts expected
- ``dnl < 1.0``: channel is narrower than nominal → fewer counts expected

A correction factor of 1.0 means the channel has nominal width.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def estimate_dnl_from_ratio(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    min_counts: float = 50.0,
    max_correction: float = 0.5,
) -> np.ndarray:
    """Estimate per-bin DNL correction from observed/predicted count ratio.

    Parameters
    ----------
    observed : array-like
        Observed counts per bin (histogram).
    predicted : array-like
        Model-predicted counts per bin (using nominal bin widths).
    min_counts : float
        Minimum predicted counts to include a bin in DNL estimation.
        Bins below this threshold get correction = 1.0.
    max_correction : float
        Maximum absolute deviation from 1.0.  The returned factors are
        clipped to ``[1 - max_correction, 1 + max_correction]``.

    Returns
    -------
    dnl : ndarray
        Per-bin DNL correction factors.  Multiply model bin widths by
        these factors to account for non-uniform ADC channel widths.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"observed shape {observed.shape} != predicted shape {predicted.shape}"
        )

    dnl = np.ones_like(observed)
    mask = predicted > min_counts
    dnl[mask] = observed[mask] / predicted[mask]

    # Clip to prevent extreme corrections
    dnl = np.clip(dnl, 1.0 - max_correction, 1.0 + max_correction)

    n_corrected = int(np.sum(mask))
    n_total = len(dnl)
    dnl_range = float(dnl[mask].max() - dnl[mask].min()) if n_corrected > 0 else 0.0
    logger.info(
        "DNL estimation: %d/%d bins corrected (%.1f%%), "
        "range [%.4f, %.4f], mean=%.4f, amplitude=%.1f%%",
        n_corrected,
        n_total,
        100.0 * n_corrected / max(n_total, 1),
        float(dnl.min()),
        float(dnl.max()),
        float(dnl.mean()),
        100.0 * dnl_range,
    )

    return dnl


def save_dnl_map(
    dnl: np.ndarray,
    path: str | Path,
    *,
    adc_channels: np.ndarray | None = None,
) -> None:
    """Save DNL correction map to a NumPy ``.npz`` file.

    Parameters
    ----------
    dnl : ndarray
        Per-bin DNL correction factors.
    path : str or Path
        Output file path (should end in ``.npz``).
    adc_channels : ndarray, optional
        Corresponding ADC channel numbers (bin centers).
    """
    path = Path(path)
    data = {"dnl": np.asarray(dnl, dtype=float)}
    if adc_channels is not None:
        data["adc_channels"] = np.asarray(adc_channels, dtype=float)
    np.savez_compressed(path, **data)
    logger.info("DNL map saved to %s (%d channels)", path, len(dnl))


def load_dnl_map(path: str | Path) -> np.ndarray:
    """Load DNL correction map from a ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        Path to ``.npz`` file created by :func:`save_dnl_map`.

    Returns
    -------
    dnl : ndarray
        Per-bin DNL correction factors.
    """
    path = Path(path)
    data = np.load(path)
    dnl = data["dnl"]
    logger.info("DNL map loaded from %s (%d channels)", path, len(dnl))
    return dnl
