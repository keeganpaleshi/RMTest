"""Utilities for handling baseline records and provenance."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from typing import Any, Iterable


def _maybe_float(value: Any) -> float | None:
    """Return ``value`` as ``float`` when possible, otherwise ``None``."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _isoformat(value: Any) -> str | None:
    """Return an ISO-8601 string for ``value`` when possible."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def _evaluate_polynomial(coeffs: Any, x: Any) -> float | None:
    """Evaluate a polynomial with coefficients ``coeffs`` at ``x``."""

    x_val = _maybe_float(x)
    if x_val is None:
        return None

    if isinstance(coeffs, Mapping):
        items = []
        for key, val in coeffs.items():
            try:
                exp = int(key)
            except (TypeError, ValueError):
                continue
            c_val = _maybe_float(val)
            if c_val is None:
                continue
            items.append((exp, c_val))
        if not items:
            return None
        items.sort()
        return float(sum(c * (x_val ** exp) for exp, c in items))

    try:
        coeff_list = [float(c) for c in coeffs]
    except (TypeError, ValueError):
        return None

    return float(sum(coef * (x_val ** idx) for idx, coef in enumerate(coeff_list)))


def _extract_calibration_energy(calibration: Any) -> tuple[float | None, float | None]:
    """Return ``(centroid_MeV, sigma_E)`` for Po214 from ``calibration``."""

    if calibration is None:
        return None, None

    sigma_val = None
    centroid_mev = None
    peaks = None

    if isinstance(calibration, Mapping):
        sigma_val = _maybe_float(calibration.get("sigma_E"))
        peaks = calibration.get("peaks")
        coeffs = calibration.get("coeffs")
    else:
        sigma_val = _maybe_float(getattr(calibration, "sigma_E", None))
        peaks = getattr(calibration, "peaks", None)
        coeffs = getattr(calibration, "coeffs", None)

    if isinstance(peaks, Mapping):
        peak = peaks.get("Po214")
        if isinstance(peak, Mapping):
            centroid_mev = _maybe_float(peak.get("centroid_energy_MeV"))
            if centroid_mev is None:
                centroid_adc = peak.get("centroid_adc")
                centroid_mev = _evaluate_polynomial(coeffs, centroid_adc)
    return centroid_mev, sigma_val


def initialize_baseline_record(
    baseline_info: MutableMapping[str, Any] | None,
    *,
    calibration: Any = None,
) -> dict[str, Any] | None:
    """Initialise and attach a structured baseline record to ``baseline_info``."""

    if not baseline_info:
        return None

    timestamp_range = None
    start = baseline_info.get("start")
    end = baseline_info.get("end")
    if start is not None or end is not None:
        timestamp_range = [_isoformat(start), _isoformat(end)]

    record: dict[str, Any] = {
        "timestamp_range": timestamp_range,
        "live_time_s": _maybe_float(baseline_info.get("live_time")) or 0.0,
        "dilution_factor": _maybe_float(baseline_info.get("dilution_factor")),
        "rates_Bq": {},
        "rate_unc_Bq": {},
        "scale_factors": {},
    }

    scales = baseline_info.get("scales")
    if isinstance(scales, Mapping):
        record["scale_factors"] = {
            str(iso): _maybe_float(val) or 0.0 for iso, val in scales.items()
        }

    centroid_mev, sigma_val = _extract_calibration_energy(calibration)
    if centroid_mev is not None:
        record["po214_centroid_MeV"] = centroid_mev
        baseline_info["po214_centroid_MeV"] = centroid_mev
    if sigma_val is not None:
        record["po214_sigma_E_MeV"] = sigma_val
        baseline_info["po214_sigma_E_MeV"] = sigma_val

    baseline_info["record"] = record
    return record


def update_record_with_counts(
    record: MutableMapping[str, Any] | None,
    isotope: str,
    counts: float,
    live_time_s: float,
    efficiency: float,
) -> tuple[float, float]:
    """Update ``record`` with rate information derived from ``counts``."""

    if not record or isotope is None:
        return 0.0, 0.0

    counts_val = _maybe_float(counts) or 0.0
    live_time = _maybe_float(live_time_s) or 0.0
    eff = _maybe_float(efficiency) or 0.0

    if live_time <= 0 or eff <= 0:
        rate = 0.0
        sigma = 0.0
    else:
        rate = counts_val / (live_time * eff)
        sigma = math.sqrt(max(counts_val, 0.0)) / (live_time * eff)

    rates = record.setdefault("rates_Bq", {})
    sigmas = record.setdefault("rate_unc_Bq", {})
    rates[isotope] = float(rate)
    sigmas[isotope] = float(sigma)
    return float(rate), float(sigma)


def finalize_baseline_record(
    record: Mapping[str, Any] | None,
    baseline_info: MutableMapping[str, Any] | None,
) -> None:
    """Copy structured baseline record information into ``baseline_info``."""

    if not record or baseline_info is None:
        return

    rates = record.get("rates_Bq")
    if isinstance(rates, Mapping) and rates:
        baseline_info["rate_Bq"] = {str(k): float(v) for k, v in rates.items()}

    sigmas = record.get("rate_unc_Bq")
    if isinstance(sigmas, Mapping) and sigmas:
        baseline_info["rate_unc_Bq"] = {str(k): float(v) for k, v in sigmas.items()}

    dilution = record.get("dilution_factor")
    if dilution is not None:
        baseline_info.setdefault("dilution_factor", float(dilution))

    scales = record.get("scale_factors")
    if isinstance(scales, Mapping) and scales:
        baseline_info.setdefault(
            "scales", {str(k): float(v) for k, v in scales.items()}
        )

    baseline_info["record"] = dict(record)


def get_fixed_background_for_time_fit(
    record: Mapping[str, Any] | None,
    isotope: str,
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return fixed-background information for ``isotope`` using ``record``."""

    if not record or isotope is None:
        return None

    rates = record.get("rates_Bq")
    if not isinstance(rates, Mapping) or isotope not in rates:
        return None

    base_rate = _maybe_float(rates.get(isotope))
    if base_rate is None:
        return None

    sigmas = record.get("rate_unc_Bq") if isinstance(record, Mapping) else {}
    base_sigma = None
    if isinstance(sigmas, Mapping):
        base_sigma = _maybe_float(sigmas.get(isotope))

    scales = record.get("scale_factors")
    scale = None
    if isinstance(scales, Mapping):
        scale = _maybe_float(scales.get(isotope))

    if scale is None:
        cfg_scale = None
        if config and isinstance(config.get("scales"), Mapping):
            cfg_scale = _maybe_float(config["scales"].get(isotope))
        scale = cfg_scale if cfg_scale is not None else 1.0

    background_rate = base_rate * scale
    background_unc = (base_sigma or 0.0) * scale if base_sigma is not None else 0.0

    result = {
        "background_rate_Bq": float(background_rate),
        "background_unc_Bq": float(background_unc),
        "mode": "fixed_from_baseline",
        "baseline_activity_Bq": float(background_rate),
        "baseline_activity_unc_Bq": float(background_unc),
        "dilution_factor": record.get("dilution_factor"),
    }

    ts_range = record.get("timestamp_range")
    if isinstance(ts_range, Iterable):
        result["source_range"] = list(ts_range)

    return result


_LEGACY_BACKGROUND_MODE_KEYS = {
    "baselinefixed",
    "baselinefrombaseline",
    "fixedfrombaseline",
}


def _normalize_background_mode(mode: str | None) -> str | None:
    """Return a canonical background mode string for reporting."""

    if mode is None:
        return None

    if not isinstance(mode, str):
        return mode

    canonical = mode.strip()
    lowered = canonical.lower().replace(" ", "")
    lowered = lowered.replace("-", "").replace("_", "")

    if lowered in _LEGACY_BACKGROUND_MODE_KEYS:
        return "fixed_from_baseline"

    return canonical


def normalize_background_mode(mode: str | None) -> str | None:
    """Public wrapper so other modules can normalize background modes."""

    return _normalize_background_mode(mode)


def apply_time_fit_provenance(
    time_fit_summary: MutableMapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]] | None,
    record: Mapping[str, Any] | None,
) -> None:
    """Attach baseline provenance to ``time_fit_summary`` entries."""

    if not provenance or not time_fit_summary:
        return

    ts_range = None
    if record is not None:
        ts_range = record.get("timestamp_range")
        if isinstance(ts_range, Iterable):
            ts_range = list(ts_range)

    for iso, prov in provenance.items():
        entry = time_fit_summary.get(iso)
        if entry is None:
            continue

        mode = prov.get("mode")
        mode = normalize_background_mode(mode) or "fixed_from_baseline"
        entry["background_source"] = mode

        range_val = prov.get("source_range")
        if not range_val and ts_range is not None:
            range_val = ts_range
        if range_val:
            entry["baseline_source_range"] = list(range_val)

        if prov.get("baseline_activity_Bq") is not None:
            entry["baseline_activity_Bq"] = float(prov["baseline_activity_Bq"])
        if prov.get("baseline_activity_unc_Bq") is not None:
            entry["baseline_activity_unc_Bq"] = float(
                prov["baseline_activity_unc_Bq"]
            )

        dilution = prov.get("dilution_factor")
        if dilution is None and record is not None:
            dilution = record.get("dilution_factor")
        if dilution is not None:
            entry["baseline_dilution_factor"] = float(dilution)


def assess_baseline_drift(
    record: Mapping[str, Any] | None,
    calibration: Any,
    baseline_config: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Return ``(flag, message)`` indicating baseline drift compatibility."""

    if not record:
        return False, None

    baseline_energy = _maybe_float(record.get("po214_centroid_MeV"))
    baseline_sigma = _maybe_float(record.get("po214_sigma_E_MeV"))

    current_energy, current_sigma = _extract_calibration_energy(calibration)

    if baseline_config is None:
        baseline_config = {}

    energy_tol = _maybe_float(baseline_config.get("drift_energy_tolerance_MeV"))
    if energy_tol is None:
        energy_tol = 0.05

    sigma_tol = _maybe_float(
        baseline_config.get("drift_sigma_fraction_tolerance")
    )
    if sigma_tol is None:
        sigma_tol = 0.2

    warn = False

    if (
        baseline_energy is not None
        and current_energy is not None
        and abs(current_energy - baseline_energy) > energy_tol
    ):
        warn = True

    if baseline_sigma is not None and current_sigma is not None and baseline_sigma != 0:
        frac_diff = abs(current_sigma - baseline_sigma) / abs(baseline_sigma)
        if frac_diff > sigma_tol:
            warn = True

    if warn:
        return True, "baseline spectral shape drifted relative to assay run"
    return False, None


__all__ = [
    "initialize_baseline_record",
    "update_record_with_counts",
    "finalize_baseline_record",
    "get_fixed_background_for_time_fit",
    "apply_time_fit_provenance",
    "assess_baseline_drift",
]
