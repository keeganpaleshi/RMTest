import numpy as np

__all__ = ["neg_loglike_extended"]


def _softplus(x):
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def neg_loglike_extended(
    E, intensity_fn, params, *, area_keys, clip=1e-300, background_model=None, domain=None
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
    domain : tuple, optional
        (Emin, Emax) defining the fit range in MeV. Required when background
        parameters b0/b1 are present to compute the background integral.

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

    # Compute total expected counts: sum of peak areas + background integral
    Nexp = float(sum(_softplus(params[k]) for k in area_keys))

    # Add background integral if b0/b1 are present and not accounted for via S_bkg
    if "b0" in params and "b1" in params and "S_bkg" not in area_keys:
        if domain is None:
            # Estimate domain from data if not provided
            if E.size > 0:
                Emin, Emax = float(E.min()), float(E.max())
            else:
                raise ValueError(
                    "domain must be provided when using b0/b1 background without S_bkg"
                )
        else:
            Emin, Emax = domain

        b0 = float(params["b0"])
        b1 = float(params["b1"])

        # Integral of (b0 + b1*E) from Emin to Emax
        background_integral = b0 * (Emax - Emin) + 0.5 * b1 * (Emax**2 - Emin**2)
        Nexp += background_integral

    return float(-(np.sum(np.log(lam)) - Nexp))
