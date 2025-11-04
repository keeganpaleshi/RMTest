import numpy as np

__all__ = ["neg_loglike_extended"]


def _softplus(x):
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def neg_loglike_extended(
    E,
    intensity_fn,
    params,
    *,
    area_keys,
    clip=1e-300,
    background_model=None,
    bounds=None,
    expected_total=None,
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
        Name of the background model. When set to ``"loglin_unit"`` required
        background parameters are validated before evaluation.
    bounds : tuple[float, float], optional
        Energy bounds ``(E_lo, E_hi)`` used when computing the expected event
        count by integrating the background model. Required unless
        ``expected_total`` is provided.
    expected_total : float, optional
        Precomputed integral of the spectral intensity over the fit window. If
        supplied the ``bounds`` argument is ignored.

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

    if background_model == "loglin_unit":
        required = {"S_bkg", "b0", "b1"}
        missing = required - params.keys()
        if missing:
            got = sorted(params.keys())
            raise ValueError(
                "background_model=loglin_unit requires params {S_bkg, b0, b1}; got: "
                f"{got}"
            )

    missing_areas = [k for k in area_keys if k not in params]
    if missing_areas:
        got = sorted(params.keys())
        needed = ", ".join(area_keys)
        raise ValueError(
            f"likelihood=extended requires params {{{needed}}}; got: {got}"
        )

    lam = np.clip(intensity_fn(E, params), clip, np.inf)

    if expected_total is None:
        if bounds is None:
            raise ValueError(
                "neg_loglike_extended requires bounds when expected_total is not provided"
            )
        E_lo, E_hi = map(float, bounds)
        peak_keys = [k for k in area_keys if k != "S_bkg"]
        total = float(sum(_softplus(params[k]) for k in peak_keys))

        if background_model == "loglin_unit":
            if "S_bkg" not in params:
                raise ValueError(
                    "background_model=loglin_unit requires S_bkg to compute expected counts"
                )
            total += float(_softplus(params["S_bkg"]))
        elif "S_bkg" in params:
            total += float(_softplus(params["S_bkg"]))
        else:
            b0 = float(params.get("b0", 0.0))
            b1 = float(params.get("b1", 0.0))
            total += b0 * (E_hi - E_lo) + 0.5 * b1 * (E_hi**2 - E_lo**2)
        Nexp = total
    else:
        Nexp = float(expected_total)

    return float(-(np.sum(np.log(lam)) - Nexp))
