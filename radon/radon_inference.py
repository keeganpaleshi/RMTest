"""Convert progeny time-series counts into inferred Rn-222 activity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import logging
import math

import numpy as np


logger = logging.getLogger(__name__)

__all__ = ["run_radon_inference"]


def _to_epoch_seconds(value: Any) -> float | None:
    """Return ``value`` expressed as Unix seconds when possible."""

    if value is None:
        return None

    if isinstance(value, np.datetime64):
        return float(value.astype("datetime64[ns]").astype("int64")) / 1e9

    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return float(value.timestamp())

    if hasattr(value, "timestamp") and not isinstance(value, (int, float, np.floating)):
        try:
            return float(value.timestamp())
        except Exception:  # pragma: no cover - safety net for unusual types
            pass

    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    positive = {iso: float(w) for iso, w in weights.items() if w is not None and w > 0}
    if not positive:
        n = len(weights)
        if n == 0:
            return {}
        equal = 1.0 / float(n)
        return {iso: equal for iso in weights}
    total = sum(positive.values())
    if total <= 0:
        return {}
    return {iso: positive.get(iso, 0.0) / total for iso in weights}


def _apply_chain_correction(activity_bq: float, mode: str, iso: str) -> float:
    mode = (mode or "none").lower()
    if mode in ("none", "assume_equilibrium"):
        return activity_bq
    logger.warning(
        "Unsupported chain_correction=%s for isotope %s; leaving activity unchanged",
        mode,
        iso,
    )
    return activity_bq


def _extract_series(series_like: Any) -> list[Mapping[str, Any]]:
    if series_like is None:
        return []
    if isinstance(series_like, Sequence) and not isinstance(series_like, (str, bytes)):
        return [entry if isinstance(entry, Mapping) else dict(entry) for entry in series_like]
    return []


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_ambient_series(
    ambient_raw: Sequence[Any] | None,
    target_times: Sequence[dict[str, Any]],
) -> list[dict[str, float | None]]:
    if not ambient_raw:
        return []

    prepared: list[tuple[float | None, float | None]] = []
    for item in ambient_raw:
        if isinstance(item, Mapping):
            t_val = _to_epoch_seconds(item.get("t") or item.get("timestamp"))
            val = _coerce_float(
                item.get("rn_bq_per_m3")
                or item.get("value")
                or item.get("ambient_bq_per_m3")
            )
        else:
            try:
                t_raw, v_raw = item
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            t_val = _to_epoch_seconds(t_raw)
            val = _coerce_float(v_raw)
        prepared.append((t_val, val))

    if not prepared:
        return []

    series: list[dict[str, float | None]] = []
    last_val: float | None = None
    for idx, rn_entry in enumerate(target_times):
        t = _coerce_float(rn_entry.get("t"))
        ambient_val: float | None = None
        if idx < len(prepared):
            candidate = prepared[idx][1]
            if candidate is not None and math.isfinite(candidate):
                last_val = candidate
                ambient_val = candidate
        if ambient_val is None:
            ambient_val = last_val
        series.append({"t": t, "rn_bq_per_m3": ambient_val})
    return series


def run_radon_inference(
    isotope_series: Mapping[str, Sequence[Mapping[str, Any]] | Sequence[Any] | Any] | None,
    config: Mapping[str, Any] | None,
    *,
    external_rn_series: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return inferred radon activity derived from progeny counts."""

    if not isinstance(config, Mapping):
        return {}

    radon_cfg = config.get("radon_inference")
    if not isinstance(radon_cfg, Mapping) or not radon_cfg.get("enabled"):
        return {}

    if not isinstance(isotope_series, Mapping):
        logger.info("Radon inference requested but no isotope time series available")
        return {}

    source_isotopes = [
        iso for iso in radon_cfg.get("source_isotopes", []) if isinstance(iso, str)
    ]
    detection_cfg = radon_cfg.get("detection_efficiency", {})
    if not isinstance(detection_cfg, Mapping):
        detection_cfg = {}

    available_series: dict[str, list[Mapping[str, Any]]] = {}
    missing_isotopes: list[str] = []
    for iso in source_isotopes:
        entries = _extract_series(isotope_series.get(iso))
        if entries:
            available_series[iso] = entries
        else:
            missing_isotopes.append(iso)

    if missing_isotopes:
        logger.warning(
            "Radon inference skipping missing isotope series: %s",
            ", ".join(sorted(missing_isotopes)),
        )

    if not available_series:
        return {
            "rn_inferred": [],
            "ambient_rn": [],
            "volume_equiv": [],
            "volume_cumulative": [],
            "meta": {
                "source_isotopes": [],
                "source_weights": {},
                "detection_efficiency": {},
                "transport_efficiency": float(radon_cfg.get("transport_efficiency", 1.0) or 1.0),
                "retention_efficiency": float(radon_cfg.get("retention_efficiency", 1.0) or 1.0),
                "chain_correction": radon_cfg.get("chain_correction", "none"),
                "missing_isotopes": missing_isotopes,
            },
        }

    raw_weights = {}
    for iso in available_series:
        weight_cfg = radon_cfg.get("source_weights", {})
        weight_val = weight_cfg.get(iso, 1.0)
        raw_weights[iso] = _coerce_float(weight_val) or 0.0
    weights_norm = _normalize_weights(raw_weights)
    if not weights_norm:
        equal = 1.0 / float(len(available_series))
        weights_norm = {iso: equal for iso in available_series}

    transport_eff = _coerce_float(radon_cfg.get("transport_efficiency", 1.0))
    if transport_eff is None or transport_eff <= 0:
        transport_eff = 1.0
    retention_eff = _coerce_float(radon_cfg.get("retention_efficiency", 1.0))
    if retention_eff is None or retention_eff <= 0:
        retention_eff = 1.0

    chain_mode = str(radon_cfg.get("chain_correction", "none"))

    base_iso = next(iter(available_series))
    base_series = available_series[base_iso]
    n_bins = len(base_series)
    if n_bins == 0:
        return {
            "rn_inferred": [],
            "ambient_rn": [],
            "volume_equiv": [],
            "volume_cumulative": [],
            "meta": {
                "source_isotopes": list(available_series.keys()),
                "source_weights": weights_norm,
                "detection_efficiency": {},
                "transport_efficiency": transport_eff,
                "retention_efficiency": retention_eff,
                "chain_correction": chain_mode,
                "missing_isotopes": missing_isotopes,
            },
        }

    rn_entries: list[dict[str, Any]] = []
    ambient_entries = _build_ambient_series(external_rn_series, base_series)
    detection_missing_logged: set[str] = set()
    volume_entries: list[dict[str, Any]] = []
    cumulative_entries: list[dict[str, Any]] = []
    cumulative_volume = 0.0

    for idx in range(n_bins):
        base_entry = base_series[idx]
        time_val = _coerce_float(base_entry.get("t"))
        dt_base = _coerce_float(base_entry.get("dt"))
        if dt_base is None or dt_base <= 0:
            continue

        contributions: dict[str, float] = {}
        weights_present: dict[str, float] = {}
        per_source_meta: dict[str, Any] = {}

        for iso, series in available_series.items():
            if idx >= len(series):
                continue
            entry = series[idx]
            counts = _coerce_float(entry.get("counts"))
            dt_iso = _coerce_float(entry.get("dt", dt_base))
            if counts is None or dt_iso is None or dt_iso <= 0:
                continue
            det_eff = _coerce_float(detection_cfg.get(iso))
            if det_eff is None or det_eff <= 0:
                if iso not in detection_missing_logged:
                    logger.warning(
                        "Radon inference missing detection_efficiency for %s", iso
                    )
                    detection_missing_logged.add(iso)
                continue

            global_eff = det_eff * transport_eff * retention_eff
            if global_eff <= 0:
                continue

            activity_iso = counts / (global_eff * dt_iso)
            activity_iso = _apply_chain_correction(activity_iso, chain_mode, iso)
            contributions[iso] = activity_iso
            weights_present[iso] = weights_norm.get(iso, 0.0)
            per_source_meta[iso] = {
                "counts": counts,
                "dt": dt_iso,
                "weight": weights_norm.get(iso, 0.0),
                "activity_bq": activity_iso,
            }

        if not contributions:
            rn_entries.append(
                {
                    "t": time_val,
                    "rn_bq": 0.0,
                    "dt": dt_base,
                    "source": ", ".join(sorted(available_series)),
                    "sources": per_source_meta,
                }
            )
            ambient_val = None
            if idx < len(ambient_entries):
                ambient_val = ambient_entries[idx].get("rn_bq_per_m3")
            volume_entries.append({"t": time_val, "v_m3": None, "v_lpm": None})
            continue

        weight_sum_present = sum(weights_present.values())
        if weight_sum_present <= 0:
            equal = 1.0 / float(len(contributions))
            weights_present = {iso: equal for iso in contributions}
            weight_sum_present = 1.0

        renorm_factor = 1.0 / weight_sum_present
        rn_value = sum(contributions[iso] * weights_present[iso] for iso in contributions)
        rn_value *= renorm_factor

        for iso in per_source_meta:
            per_source_meta[iso]["weight_effective"] = weights_present[iso] * renorm_factor

        rn_entries.append(
            {
                "t": time_val,
                "rn_bq": rn_value,
                "dt": dt_base,
                "source": ", ".join(sorted(contributions)),
                "sources": per_source_meta,
            }
        )

        ambient_val = None
        if idx < len(ambient_entries):
            ambient_val = ambient_entries[idx].get("rn_bq_per_m3")
        volume_entry: dict[str, Any]
        if ambient_val is None or ambient_val <= 0:
            volume_entry = {"t": time_val, "v_m3": None, "v_lpm": None}
        else:
            volume_m3 = rn_value * dt_base / ambient_val
            minutes = dt_base / 60.0 if dt_base > 0 else None
            v_lpm = volume_m3 * 1000.0 / minutes if minutes and minutes > 0 else None
            volume_entry = {"t": time_val, "v_m3": volume_m3, "v_lpm": v_lpm}
            cumulative_volume += volume_m3
            cumulative_entries.append({"t": time_val, "v_m3_cum": cumulative_volume})
        volume_entries.append(volume_entry)

    meta = {
        "source_isotopes": list(available_series.keys()),
        "source_weights": weights_norm,
        "detection_efficiency": {
            iso: _coerce_float(detection_cfg.get(iso)) for iso in available_series
        },
        "transport_efficiency": transport_eff,
        "retention_efficiency": retention_eff,
        "chain_correction": chain_mode,
    }
    if missing_isotopes:
        meta["missing_isotopes"] = missing_isotopes

    return {
        "rn_inferred": rn_entries,
        "ambient_rn": [
            {"t": entry.get("t"), "rn_bq_per_m3": entry.get("rn_bq_per_m3")}
            for entry in ambient_entries
        ],
        "volume_equiv": volume_entries,
        "volume_cumulative": cumulative_entries,
        "meta": meta,
    }

