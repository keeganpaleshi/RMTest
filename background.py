import numpy as np

__all__ = [
    "estimate_linear_background",
    "estimate_polynomial_background",
    "estimate_polynomial_background_auto",
]


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
        The fit is weighted by ``1/sqrt(counts)`` assuming Poisson statistics.

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
    counts = hist.astype(float)
    weights = np.zeros_like(counts, dtype=float)
    positive = counts > 0
    weights[positive] = 1.0 / np.sqrt(counts[positive])
    mask = np.ones_like(centers, dtype=bool)
    for mu in centroids.values():
        mask &= ~((centers >= mu - peak_width) & (centers <= mu + peak_width))

    if mask.sum() < 2 or not np.any(weights[mask] > 0):
        return 0.0, 0.0

    coeffs = np.polyfit(centers[mask], hist[mask], 1, w=weights[mask])
    b1, b0 = coeffs
    return float(b0), float(b1)


def estimate_polynomial_background(
    energies,
    centroids,
    degree,
    peak_width=0.3,
    bins="fd",
):
    """Estimate polynomial continuum coefficients.

    Parameters
    ----------
    energies : array-like
        Energy values in MeV.
    centroids : dict
        Mapping of peak name to centroid energy.
    degree : int
        Polynomial degree. 0 yields a constant background.
    peak_width : float, optional
        Half-width around each centroid to exclude from the fit.
    bins : int or sequence or str, optional
        Histogram bin specification passed to ``numpy.histogram``.

    Returns
    -------
    ndarray
        Polynomial coefficients ``[b0, b1, ...]`` in ascending order.
    """
    e = np.asarray(energies, dtype=float)
    if e.size == 0:
        return np.zeros(degree + 1)

    hist, edges = np.histogram(e, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = hist.astype(float)
    weights = np.zeros_like(counts, dtype=float)
    positive = counts > 0
    weights[positive] = 1.0 / np.sqrt(counts[positive])
    mask = np.ones_like(centers, dtype=bool)
    for mu in centroids.values():
        mask &= ~((centers >= mu - peak_width) & (centers <= mu + peak_width))

    if mask.sum() <= degree or not np.any(weights[mask] > 0):
        return np.zeros(degree + 1)

    coeffs = np.polyfit(centers[mask], hist[mask], degree, w=weights[mask])
    return coeffs[::-1]


def estimate_polynomial_background_auto(
    energies,
    centroids,
    max_order=2,
    peak_width=0.3,
    bins="fd",
):
    """Estimate a polynomial continuum with order selected by AIC.

    Parameters
    ----------
    max_order : int
        Maximum polynomial degree to try.

    Returns
    -------
    tuple
        ``(coeffs, order)`` with coefficients in ascending order and the chosen
        degree.
    """
    best_coeffs = np.zeros(1)
    best_order = 0
    best_aic = np.inf
    aic_list = []

    hist, edges = np.histogram(np.asarray(energies, dtype=float), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = hist.astype(float)
    weights = np.zeros_like(counts, dtype=float)
    positive = counts > 0
    weights[positive] = 1.0 / np.sqrt(counts[positive])
    mask = np.ones_like(centers, dtype=bool)
    for mu in centroids.values():
        mask &= ~((centers >= mu - peak_width) & (centers <= mu + peak_width))

    x = centers[mask]
    y = hist[mask]
    w = weights[mask]

    if x.size == 0 or not np.any(w > 0):
        return best_coeffs, best_order

    for n in range(max_order + 1):
        if x.size <= n or not np.any(w > 0):
            break
        coeffs = np.polyfit(x, y, n, w=w)
        y_pred = np.polyval(coeffs, x)
        resid = (y - y_pred) * w
        chi2 = float(np.sum(resid**2))
        k = n + 1
        aic = chi2 + 2 * k
        aic_list.append((n, aic, coeffs[::-1]))
        if aic < best_aic:
            best_aic = aic
            best_coeffs = coeffs[::-1]
            best_order = n

    # Prefer the lowest order within 2 AIC of the best fit
    if aic_list:
        aic_list.sort(key=lambda t: t[0])
        best_aic = min(a[1] for a in aic_list)
        for n, aic, coeffs in aic_list:
            if aic - best_aic <= 2:
                best_order = n
                best_coeffs = coeffs
                break

    return best_coeffs, best_order
