import numpy as np

__all__ = ["neg_loglike_extended"]


def _softplus(x):
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def neg_loglike_extended(
    E, intensity_fn, params, *, area_keys, clip=1e-300, background_model=None
):
    """Extended unbinned negative log-likelihood.

    Parameters
    ----------
    E : array-like
        Unbinned energy samples in MeV.
    intensity_fn : callable
        Function mapping ``(E, params)`` to the expected intensity ``λ(E)``
        in counts/MeV.
    params : Mapping
        Parameter dictionary. Expected counts (areas) are taken from
        ``params`` via ``area_keys`` and transformed with ``softplus`` to ensure
        positivity.
    area_keys : Sequence[str]
        Keys in ``params`` corresponding to event counts.
    clip : float, optional
        Lower bound for the intensity to avoid ``log(0)``. Default is ``1e-300``.
    background_model : str, optional
        Name of the background model. When set to ``"loglin_unit"`` an
        additional background area parameter ``S_bkg`` must be supplied in
        ``params``.

    Returns
    -------
    float
        The negative log-likelihood value.

    Examples
    --------
    >>> import numpy as np
    >>> def reference_intensity(E, p):
    ...     # simple constant intensity background
    ...     return np.full_like(E, p["rate"])
    >>> E = np.array([1.0, 2.0, 3.0])
    >>> params = {"rate": 0.5, "area": 0.0}
    >>> neg_loglike_extended(E, reference_intensity, params, area_keys=("area",))
    1.5...
    """
    E = np.asarray(E, dtype=float)

    if background_model == "loglin_unit" and "S_bkg" not in params:
        got = sorted(params.keys())
        raise ValueError(
            "background_model=loglin_unit requires param S_bkg; got: " f"{got}"
        )

    missing_areas = [k for k in area_keys if k not in params]
    if missing_areas:
        got = sorted(params.keys())
        needed = ", ".join(area_keys)
        raise ValueError(
            f"likelihood=extended requires params {{{needed}}}; got: {got}"
        )

    lam = np.clip(intensity_fn(E, params), clip, np.inf)
    Nexp = float(sum(_softplus(params[k]) for k in area_keys))
    return float(-(np.sum(np.log(lam)) - Nexp))
