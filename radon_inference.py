"""Convert isotope count time-series into inferred radon activity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Any, Iterable, Mapping

from utils.time_utils import parse_timestamp


logger = logging.getLogger(__name__)


@dataclass
class _BinEntry:
    """Internal representation of a single isotope time-bin."""

    timestamp: float
    timestamp_label: str
    counts: float
    live_time: float


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_bins(entries: Iterable[Mapping[str, Any]]) -> list[_BinEntry]:
    bins: list[_BinEntry] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        ts_val = entry.get("t") or entry.get("timestamp") or entry.get("time")
        if ts_val is None:
            continue
        try:
            ts = parse_timestamp(ts_val)
        except Exception:
            logger.debug("radon_inference: could not parse timestamp %r", ts_val)
            continue

        counts_val = _coerce_float(entry.get("counts"))
        if counts_val is None:
            continue

        dt_val = (
            _coerce_float(entry.get("dt"))
            or _coerce_float(entry.get("live_time"))
            or _coerce_float(entry.get("width_s"))
        )
        if dt_val is None or dt_val <= 0:
            continue

        bins.append(
            _BinEntry(
                timestamp=ts.timestamp(),
                timestamp_label=ts.isoformat().replace("+00:00", "Z"),
                counts=counts_val,
                live_time=dt_val,
            )
        )
    bins.sort(key=lambda entry: entry.timestamp)
    return bins


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        return {iso: 0.0 for iso in weights}
    return {iso: float(w) / total for iso, w in weights.items() if w > 0}


def _build_external_map(external_series: Iterable[Any] | None) -> dict[str, float]:
    ambient: dict[str, float] = {}
    if not external_series:
        return ambient

    for item in external_series:
        if isinstance(item, Mapping):
            ts_val = item.get("t") or item.get("timestamp")
            value = item.get("rn_bq_per_m3") or item.get("value")
        else:
            try:
                ts_val, value = item  # type: ignore[misc]
            except (TypeError, ValueError):
                continue
        if ts_val is None or value is None:
            continue
        try:
            ts = parse_timestamp(ts_val)
        except Exception:
            continue
        val = _coerce_float(value)
        if val is None:
            continue
        ambient[ts.isoformat().replace("+00:00", "Z")] = val
    return ambient


def run_radon_inference(
    isotope_series: Mapping[str, Iterable[Mapping[str, Any]]],
    config: Mapping[str, Any],
    external_rn_series: Iterable[Any] | None = None,
) -> dict[str, Any] | None:
    """Return inferred radon activity and equivalent volume series."""

    radon_cfg = config.get("radon_inference")
    if not isinstance(radon_cfg, Mapping):
        return None
    if not bool(radon_cfg.get("enabled", False)):
        return None

    source_isotopes = radon_cfg.get("source_isotopes") or []
    if not isinstance(source_isotopes, list) or not source_isotopes:
        logger.warning("radon_inference: no source_isotopes configured")
        return None

    detection_cfg = radon_cfg.get("detection_efficiency") or {}
    if not isinstance(detection_cfg, Mapping):
        logger.warning("radon_inference: detection_efficiency missing or invalid")
        return None

    weight_cfg = radon_cfg.get("source_weights") or {}
    if not isinstance(weight_cfg, Mapping):
        weight_cfg = {}

    transport_eff = _coerce_float(radon_cfg.get("transport_efficiency"))
    if transport_eff is None or transport_eff <= 0:
        transport_eff = 1.0

    retention_eff = _coerce_float(radon_cfg.get("retention_efficiency"))
    if retention_eff is None or retention_eff <= 0:
        retention_eff = 1.0

    chain_correction = str(radon_cfg.get("chain_correction", "none")).lower()
    if chain_correction not in {"none", "assume_equilibrium", "forward_model"}:
        logger.warning("radon_inference: unsupported chain_correction %r", chain_correction)
        chain_correction = "none"

    available_bins: dict[str, list[_BinEntry]] = {}
    requested_weights: dict[str, float] = {}

    for iso in source_isotopes:
        weight_val = _coerce_float(weight_cfg.get(iso))
        requested_weights[iso] = weight_val if weight_val is not None else 1.0
        series_entries = isotope_series.get(iso)
        if not series_entries:
            logger.warning("radon_inference: missing isotope series for %s", iso)
            continue
        bins = _extract_bins(series_entries)
        if bins:
            available_bins[iso] = bins
        else:
            logger.warning("radon_inference: empty bins for isotope %s", iso)

    if not available_bins:
        logger.warning("radon_inference: no isotope data available after filtering")
        return None

    resolved_weights = {
        iso: requested_weights.get(iso, 1.0) for iso in available_bins.keys()
    }
    norm_weights = _normalize_weights(resolved_weights)

    detection_map: dict[str, float] = {}
    for iso in available_bins:
        det_val = _coerce_float(detection_cfg.get(iso))
        if det_val is None or det_val <= 0:
            logger.warning(
                "radon_inference: invalid detection efficiency for %s", iso
            )
            continue
        detection_map[iso] = det_val

    if not detection_map:
        logger.warning("radon_inference: no valid detection efficiencies available")
        return None

    ambient_map = _build_external_map(external_rn_series)

    per_time_counts: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    timestamps_seconds: dict[str, float] = {}

    for iso, bins in available_bins.items():
        det_eff = detection_map.get(iso)
        if det_eff is None or det_eff <= 0:
            continue
        iso_map = per_time_counts[iso]
        for bin_entry in bins:
            ts_key = bin_entry.timestamp_label
            timestamps_seconds.setdefault(ts_key, bin_entry.timestamp)
            bucket = iso_map.get(ts_key)
            if bucket is None:
                bucket = {"counts": 0.0, "live_time": 0.0}
                iso_map[ts_key] = bucket
            bucket["counts"] += bin_entry.counts
            bucket["live_time"] += bin_entry.live_time

    if not per_time_counts:
        logger.warning("radon_inference: no per-time counts accumulated")
        return None

    sorted_times = [
        key for key, _ in sorted(timestamps_seconds.items(), key=lambda kv: kv[1])
    ]

    inferred_series: list[dict[str, Any]] = []
    volume_series: list[dict[str, Any]] = []
    volume_cumulative: list[dict[str, Any]] = []
    ambient_series: list[dict[str, Any]] = []

    cumulative_volume = 0.0

    for ts_key in sorted_times:
        isotopes_present = [
            iso
            for iso, iso_map in per_time_counts.items()
            if ts_key in iso_map and detection_map.get(iso, 0) > 0
        ]
        if not isotopes_present:
            continue

        weights_for_time = {
            iso: norm_weights.get(iso, 0.0) for iso in isotopes_present
        }
        weights_for_time = {
            iso: w for iso, w in weights_for_time.items() if w > 0
        }
        if not weights_for_time:
            continue

        weights_for_time = _normalize_weights(weights_for_time)

        activities: dict[str, float] = {}
        dt_candidates: list[float] = []
        for iso in isotopes_present:
            bucket = per_time_counts[iso][ts_key]
            det_eff = detection_map[iso]
            counts_val = bucket["counts"]
            live_time = bucket["live_time"]
            if live_time <= 0 or det_eff <= 0:
                continue
            activity_iso = counts_val / (det_eff * live_time)
            activity_iso /= transport_eff * retention_eff
            activities[iso] = activity_iso
            dt_candidates.append(live_time)

        if not activities:
            continue

        dt_val = max(dt_candidates) if dt_candidates else 0.0
        rn_activity = sum(
            weights_for_time[iso] * activities[iso]
            for iso in activities
            if iso in weights_for_time
        )

        inferred_series.append(
            {
                "t": ts_key,
                "rn_bq": rn_activity,
                "dt": dt_val,
                "meta": {
                    "weights": weights_for_time,
                    "isotopes_used": sorted(activities.keys()),
                },
            }
        )

        ambient_val = ambient_map.get(ts_key)
        if ambient_val is not None and ambient_val > 0 and dt_val > 0:
            volume_equiv = (rn_activity * dt_val) / ambient_val
            minutes = dt_val / 60.0 if dt_val > 0 else 0.0
            volume_lpm = volume_equiv * 1000.0 / minutes if minutes > 0 else None
            cumulative_volume += volume_equiv
            volume_series.append(
                {
                    "t": ts_key,
                    "v_m3": volume_equiv,
                    "v_lpm": volume_lpm,
                    "ambient_bq_per_m3": ambient_val,
                }
            )
            volume_cumulative.append(
                {
                    "t": ts_key,
                    "v_m3_cum": cumulative_volume,
                }
            )
        else:
            # Preserve ordering even without ambient data
            volume_series.append(
                {
                    "t": ts_key,
                    "v_m3": None,
                    "v_lpm": None,
                    "ambient_bq_per_m3": ambient_val,
                }
            )
            volume_cumulative.append({"t": ts_key, "v_m3_cum": cumulative_volume})

    for ts_key in sorted_times:
        if ts_key in ambient_map:
            ambient_series.append(
                {"t": ts_key, "rn_bq_per_m3": ambient_map[ts_key]}
            )

    result = {
        "rn_inferred": inferred_series,
        "ambient_rn": ambient_series,
        "volume_equiv": volume_series,
        "volume_cumulative": volume_cumulative,
        "meta": {
            "source_isotopes": list(source_isotopes),
            "available_isotopes": sorted(available_bins.keys()),
            "source_weights": {
                iso: requested_weights.get(iso, 1.0) for iso in source_isotopes
            },
            "resolved_source_weights": norm_weights,
            "detection_efficiency": {
                iso: detection_map.get(iso)
                for iso in sorted(detection_map.keys())
            },
            "transport_efficiency": transport_eff,
            "retention_efficiency": retention_eff,
            "chain_correction": chain_correction,
        },
    }

    if ambient_map:
        result["meta"]["external_rn_mode"] = (
            str((radon_cfg.get("external_rn") or {}).get("mode", "constant")).lower()
        )

    return result


__all__ = ["run_radon_inference"]

