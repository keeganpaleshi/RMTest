"""Helpers for deciding when to enable EMG tails in spectral fits.

This module centralises the slightly fiddly precedence rules that govern
whether an exponentially modified Gaussian (EMG) tail should be enabled for a
given isotope and, if so, which initial tau value should seed the optimiser.
Historically this logic was scattered between configuration loading and the
fitting routines themselves which made it easy for the two paths to drift
apart.  The :func:`resolve_emg_usage` helper defined here offers a single entry
point that mirrors the behaviour exercised throughout the existing codebase:

* An explicit tau prior forces the EMG tail on.
* Per-isotope overrides supplied via ``flags['use_emg']`` (or the attribute
  equivalent) take precedence next.
* A scalar ``flags['use_emg']`` toggles all isotopes that are not otherwise
  forced on.
* Optional tau defaults may be supplied either as a scalar or mapping.  These
  are only applied when the caller enables an EMG tail without providing an
  explicit prior.

The helper returns a mapping of isotope name to :class:`EMGTailSpec` describing
the resolved state.  Callers can use the ``enabled`` flag to decide whether to
include an EMG component and may optionally read the ``mean``/``sigma`` pair to
bootstrap a tau prior when none was provided upstream.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

__all__ = ["EMGTailSpec", "resolve_emg_usage"]


@dataclass(frozen=True)
class EMGTailSpec:
    """Resolved EMG tail configuration for a single isotope."""

    enabled: bool
    mean: float | None = None
    sigma: float | None = None
    source: str = "default"

    def as_prior(self, default_sigma: float = 1.0) -> tuple[float, float] | None:
        """Return ``(mean, sigma)`` suitable for use as a prior tuple."""

        if not self.enabled or self.mean is None:
            return None
        sigma = self.sigma if self.sigma is not None else default_sigma
        sigma_val = float(abs(sigma)) if sigma is not None else float(default_sigma)
        return float(self.mean), sigma_val


def _coerce_mapping(value: Mapping[str, Any] | SimpleNamespace | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return vars(value)


def _get_flag(flags: Mapping[str, Any] | SimpleNamespace | None, key: str) -> Any:
    if flags is None:
        return None
    if isinstance(flags, Mapping):
        return flags.get(key)
    return getattr(flags, key, None)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _coerce_tau_pair(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return (None, None)
    if isinstance(value, Mapping):
        return _coerce_float(value.get("mean")), _coerce_float(value.get("sigma"))
    if isinstance(value, SimpleNamespace):  # pragma: no cover - symmetry with Mapping
        return _coerce_float(getattr(value, "mean", None)), _coerce_float(
            getattr(value, "sigma", None)
        )
    if isinstance(value, (tuple, list)) and value:
        mean = _coerce_float(value[0])
        sigma = _coerce_float(value[1]) if len(value) > 1 else None
        return mean, sigma
    return _coerce_float(value), None


def _prepare_tau_defaults(
    isotopes: Iterable[str],
    tau_defaults: Any,
) -> dict[str, tuple[float | None, float | None]]:
    isotopes = list(isotopes)
    defaults: dict[str, tuple[float | None, float | None]] = {
        iso: (None, None) for iso in isotopes
    }
    if tau_defaults is None:
        return defaults
    if isinstance(tau_defaults, (Mapping, SimpleNamespace)):
        data = _coerce_mapping(tau_defaults)
        for iso in isotopes:
            if iso in data:
                defaults[iso] = _coerce_tau_pair(data[iso])
        return defaults
    mean, sigma = _coerce_tau_pair(tau_defaults)
    if mean is None and sigma is None:
        return defaults
    for iso in isotopes:
        defaults[iso] = (mean, sigma)
    return defaults


def resolve_emg_usage(
    isotopes: Iterable[str],
    priors: Mapping[str, Any] | None,
    *,
    flags: Mapping[str, Any] | SimpleNamespace | None = None,
    tau_defaults: Any = None,
    tau_floor: float | None = None,
) -> dict[str, EMGTailSpec]:
    """Resolve EMG usage for ``isotopes`` returning an :class:`EMGTailSpec` map."""

    priors = priors or {}
    isotopes = list(isotopes)
    tau_defaults = _prepare_tau_defaults(
        isotopes,
        tau_defaults
        if tau_defaults is not None
        else _get_flag(flags, "emg_tau")
        or _get_flag(flags, "emg_tau_defaults")
        or _get_flag(flags, "emg_tau_default"),
    )

    forced_true: set[str] = set()
    prior_pairs: dict[str, tuple[float | None, float | None]] = {}
    for iso in isotopes:
        key = f"tau_{iso}"
        if key in priors:
            mean, sigma = _coerce_tau_pair(priors[key])
            prior_pairs[iso] = (mean, sigma)
            forced_true.add(iso)

    use_emg_flag = _get_flag(flags, "use_emg")
    result: dict[str, EMGTailSpec] = {}

    for iso in isotopes:
        enabled = iso in forced_true
        reason = "prior" if enabled else "default"
        mean, sigma = prior_pairs.get(iso, (None, None))
        if not enabled and use_emg_flag is not None:
            override: Any
            if isinstance(use_emg_flag, (Mapping, SimpleNamespace)):
                data = _coerce_mapping(use_emg_flag)
                override = data.get(iso)
            else:
                override = use_emg_flag
            if override is not None:
                enabled = bool(override)
                reason = "flag"
        if enabled:
            if mean is None:
                mean, sigma = tau_defaults.get(iso, (None, None))
                if mean is not None and reason == "default":
                    reason = "fallback"
            if tau_floor is not None and mean is not None:
                mean = max(mean, tau_floor)
            result[iso] = EMGTailSpec(True, mean, sigma, reason)
        else:
            result[iso] = EMGTailSpec(False, None, None, "disabled")
    return result
