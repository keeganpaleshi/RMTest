"""Utilities for working with baseline records and provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, MutableMapping

from baseline_utils import compute_dilution_factor


Number = float | int


@dataclass(frozen=True)
class FixedBackground:
    """Container for baseline-derived background information."""

    rate_Bq: float
    uncertainty_Bq: float
    mode: str = "baseline_fixed"


def _to_iso(value: Any) -> str | None:
    """Return ISO-8601 string for ``value`` when possible."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=None).isoformat()  # type: ignore[return-value]
        return value.isoformat()
    if isinstance(value, str):
        return value
    try:
        # ``pandas.Timestamp`` and ``numpy.datetime64`` both provide ``isoformat``.
        return value.isoformat()  # type: ignore[return-value]
    except AttributeError:
        return None


def _is_finite(value: Any) -> bool:
    try:
        return float(value) == float(value) and float(value) not in (float("inf"), float("-inf"))
    except (TypeError, ValueError):
        return False


def build_baseline_record(
    *,
    start: Any | None = None,
    end: Any | None = None,
    live_time_s: Number | None = None,
    dilution_factor: Number | None = None,
    intrinsic_rates_Bq: Mapping[str, Number] | None = None,
    intrinsic_unc_Bq: Mapping[str, Number] | None = None,
    calibration: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any]:
    """Return a serialisable baseline record."""

    record: dict[str, Any] = {
        "timestamp_range": [val for val in (_to_iso(start), _to_iso(end)) if val is not None],
        "intrinsic_rates_Bq": {},
    }

    if live_time_s is not None and _is_finite(live_time_s):
        record["live_time_s"] = float(live_time_s)
    if dilution_factor is not None and _is_finite(dilution_factor):
        record["dilution_factor"] = float(dilution_factor)

    if intrinsic_rates_Bq:
        for iso, rate in intrinsic_rates_Bq.items():
            if not _is_finite(rate):
                continue
            unc = None
            if intrinsic_unc_Bq:
                unc = intrinsic_unc_Bq.get(iso)
            entry = {"rate_Bq": float(rate)}
            if _is_finite(unc):
                entry["unc_Bq"] = float(unc)  # type: ignore[arg-type]
            record["intrinsic_rates_Bq"][iso] = entry

    if calibration:
        peaks = None
        sigma_E = None
        if isinstance(calibration, Mapping):
            peaks = calibration.get("peaks")
            sigma_E = calibration.get("sigma_E")
        else:
            peaks = getattr(calibration, "peaks", None)
            sigma_E = getattr(calibration, "sigma_E", None)
        if isinstance(peaks, Mapping):
            po214 = peaks.get("Po214", {})
            if isinstance(po214, Mapping):
                centroid = po214.get("centroid_mev") or po214.get("centroid_MeV")
                if _is_finite(centroid):
                    record["po214_centroid_mev"] = float(centroid)
                sigma = po214.get("sigma_E") or po214.get("sigma_mev")
                if _is_finite(sigma):
                    record["po214_sigma_E_mev"] = float(sigma)
        if _is_finite(sigma_E):
            record["sigma_E_mev"] = float(sigma_E)

    return record


def set_intrinsic_activity(
    record: MutableMapping[str, Any] | None,
    isotope: str,
    rate_Bq: Number | None,
    uncertainty_Bq: Number | None,
) -> None:
    """Update ``record`` with intrinsic activity for ``isotope``."""

    if record is None or not _is_finite(rate_Bq):
        return
    intrinsic = record.setdefault("intrinsic_rates_Bq", {})
    if not isinstance(intrinsic, dict):
        intrinsic = {}
        record["intrinsic_rates_Bq"] = intrinsic
    entry = {"rate_Bq": float(rate_Bq)}
    if _is_finite(uncertainty_Bq):
        entry["unc_Bq"] = abs(float(uncertainty_Bq))
    intrinsic[isotope] = entry


def set_scale(record: MutableMapping[str, Any] | None, isotope: str, scale: Number | None) -> None:
    """Attach scaling factor for ``isotope`` to ``record`` if finite."""

    if record is None or not _is_finite(scale):
        return
    scales = record.setdefault("scales", {})
    if not isinstance(scales, dict):
        scales = {}
        record["scales"] = scales
    scales[isotope] = float(scale)


def set_dilution_factor(record: MutableMapping[str, Any] | None, value: Number | None) -> None:
    """Store ``value`` as the dilution factor in ``record`` when finite."""

    if record is None or not _is_finite(value):
        return
    record["dilution_factor"] = float(value)


def _resolve_scale(
    record: Mapping[str, Any] | None,
    isotope: str,
    config: Mapping[str, Any] | None,
) -> float:
    if record is None:
        return 1.0

    scales = record.get("scales") if isinstance(record, Mapping) else None
    if isinstance(scales, Mapping):
        val = scales.get(isotope)
        if _is_finite(val):
            return float(val)  # type: ignore[arg-type]

    if isotope in ("Po214", "Po218"):
        dil = record.get("dilution_factor") if isinstance(record, Mapping) else None
        if _is_finite(dil):
            return float(dil)  # type: ignore[arg-type]

        baseline_cfg = config.get("baseline", {}) if isinstance(config, Mapping) else {}
        monitor = baseline_cfg.get("monitor_volume_l")
        sample = baseline_cfg.get("sample_volume_l")
        try:
            return compute_dilution_factor(monitor, sample)
        except Exception:
            pass

    return 1.0


def get_fixed_background_for_time_fit(
    baseline_record: Mapping[str, Any] | None,
    isotope: str,
    config: Mapping[str, Any] | None,
) -> FixedBackground:
    """Return fixed background derived from ``baseline_record`` for ``isotope``."""

    if not isinstance(baseline_record, Mapping):
        raise ValueError("baseline_record must be a mapping")

    intrinsic = baseline_record.get("intrinsic_rates_Bq")
    if not isinstance(intrinsic, Mapping):
        raise KeyError(f"No intrinsic rates stored for {isotope}")

    entry = intrinsic.get(isotope)
    if not isinstance(entry, Mapping):
        raise KeyError(f"No intrinsic rate for isotope {isotope}")

    rate = entry.get("rate_Bq")
    if not _is_finite(rate):
        raise ValueError(f"Invalid intrinsic rate for {isotope}")

    unc = entry.get("unc_Bq", 0.0)
    scale = _resolve_scale(baseline_record, isotope, config)
    return FixedBackground(float(rate) * scale, abs(float(unc)) * scale, "baseline_fixed")


def _extract_po214_shape(record: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(record, Mapping):
        return None, None
    centroid = record.get("po214_centroid_mev")
    sigma = record.get("po214_sigma_E_mev")
    if not _is_finite(centroid):
        centroid = None
    if not _is_finite(sigma):
        sigma = None
    return (centroid, sigma)


def _extract_current_shape(calibration: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(calibration, Mapping):
        return None, None
    peaks = calibration.get("peaks")
    centroid = None
    sigma = None
    if isinstance(peaks, Mapping):
        po214 = peaks.get("Po214", {})
        if isinstance(po214, Mapping):
            centroid = po214.get("centroid_mev") or po214.get("centroid_MeV")
    sigma = calibration.get("sigma_E") or calibration.get("sigma_E_mev")
    centroid = float(centroid) if _is_finite(centroid) else None
    sigma = float(sigma) if _is_finite(sigma) else None
    return centroid, sigma


def evaluate_baseline_drift(
    baseline_record: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    *,
    energy_tol_mev: float = 0.05,
    sigma_tol_fraction: float = 0.2,
) -> tuple[bool, str | None]:
    """Return ``(flag, message)`` describing spectral drift."""

    baseline_centroid, baseline_sigma = _extract_po214_shape(baseline_record)
    current_centroid, current_sigma = _extract_current_shape(calibration)

    if baseline_centroid is None or current_centroid is None:
        return False, None

    delta = abs(current_centroid - baseline_centroid)
    warn_energy = delta > energy_tol_mev

    warn_sigma = False
    ratio = None
    if baseline_sigma and current_sigma:
        ratio = abs(current_sigma / baseline_sigma - 1.0)
        warn_sigma = ratio > sigma_tol_fraction

    if not warn_energy and not warn_sigma:
        return False, None

    parts = [
        f"Po214 centroid drift {delta:.3f} MeV (baseline {baseline_centroid:.3f}, current {current_centroid:.3f})"
    ]
    if ratio is not None:
        parts.append(f"sigma_E change {ratio * 100:.1f}%")
    message = ", ".join(parts)
    return True, message


def annotate_summary_with_baseline(
    summary: MutableMapping[str, Any],
    baseline_record: Mapping[str, Any] | None,
    background_meta: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Attach baseline provenance details to ``summary`` in-place."""

    if baseline_record and isinstance(summary.get("baseline"), Mapping):
        summary["baseline"] = dict(summary["baseline"], record=baseline_record)
    elif baseline_record:
        summary["baseline"] = {"record": baseline_record}

    range_vals = None
    if isinstance(baseline_record, Mapping):
        timestamp_range = baseline_record.get("timestamp_range")
        if isinstance(timestamp_range, (list, tuple)):
            range_vals = [str(v) for v in timestamp_range]

    if not isinstance(background_meta, Mapping):
        return

    time_fit = summary.get("time_fit")
    if not isinstance(time_fit, Mapping):
        return

    for iso, meta in background_meta.items():
        entry = time_fit.get(iso)
        if not isinstance(entry, dict):
            continue
        mode = meta.get("mode")
        if mode:
            entry["background_source"] = mode
        if mode == "baseline_fixed":
            if range_vals:
                entry["baseline_source_range"] = list(range_vals)
            if meta.get("baseline_rate_Bq") is not None:
                entry["baseline_activity_Bq"] = float(meta["baseline_rate_Bq"])
            if meta.get("baseline_unc_Bq") is not None:
                entry["baseline_activity_unc_Bq"] = float(meta["baseline_unc_Bq"])

    radon = summary.get("radon")
    if isinstance(radon, dict):
        po214_meta = background_meta.get("Po214") if isinstance(background_meta, Mapping) else None
        if po214_meta:
            mode = po214_meta.get("mode")
            if mode:
                radon.setdefault("background_source", mode)
            if mode == "baseline_fixed" and range_vals:
                radon.setdefault("baseline_source_range", list(range_vals))


def record_baseline_drift_warning(
    summary: MutableMapping[str, Any],
    baseline_record: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    *,
    energy_tol_mev: float = 0.05,
    sigma_tol_fraction: float = 0.2,
) -> None:
    """Update ``summary`` diagnostics with baseline drift warnings."""

    warn, message = evaluate_baseline_drift(
        baseline_record,
        calibration,
        energy_tol_mev=energy_tol_mev,
        sigma_tol_fraction=sigma_tol_fraction,
    )
    if not warn:
        return

    diagnostics = summary.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics["baseline_compat_warning"] = True
    if message:
        diagnostics["baseline_compat_message"] = message
    summary["diagnostics"] = diagnostics
