from __future__ import annotations

from typing import Any, Mapping, Dict


def apply_width_prior(priors: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``priors`` applying ``spectral_fit.no_width_prior``.

    Parameters
    ----------
    priors:
        Mapping of prior parameters. ``priors`` is not modified.
    cfg:
        Configuration mapping that may contain ``spectral_fit.no_width_prior``.

    Returns
    -------
    dict
        Copy of ``priors`` with ``sigma0`` and ``F`` removed when
        ``spectral_fit.no_width_prior`` is ``True``.
    """

    out = dict(priors)
    if cfg.get("spectral_fit", {}).get("no_width_prior"):
        out.pop("sigma0", None)
        out.pop("F", None)
    return out
