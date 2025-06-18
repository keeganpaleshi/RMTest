import numpy as np

__all__ = ["estimate_linear_background"]


def estimate_linear_background(energies, centroids, peak_width=0.3, bins="fd"):
    """Estimate linear continuum parameters excluding peak regions.

    Parameters
    ----------
    energies : array-like
        Energy values in MeV.
    centroids : dict
        Mapping of peak name to centroid energy.
    peak_width : float, optional
        Half-width around each centroid to exclude from the fit.
    bins : int or sequence or str, optional
        Histogram bin specification passed to ``numpy.histogram``.
        The fit is weighted by ``sqrt(counts)`` assuming Poisson statistics.

    Returns
    -------
    tuple
        Intercept ``b0`` and slope ``b1`` of the linear background in
        counts per bin.
    """
    e = np.asarray(energies, dtype=float)
    if e.size == 0:
        return 0.0, 0.0

    hist, edges = np.histogram(e, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.sqrt(np.clip(hist, 1, None))
    mask = np.ones_like(centers, dtype=bool)
    for mu in centroids.values():
        mask &= ~((centers >= mu - peak_width) & (centers <= mu + peak_width))

    if mask.sum() < 2:
        return 0.0, 0.0

    coeffs = np.polyfit(centers[mask], hist[mask], 1, w=weights[mask])
    b1, b0 = coeffs
    return float(b0), float(b1)
