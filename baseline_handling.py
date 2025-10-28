"""Utilities for ingesting and applying baseline monitor records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import math

import pandas as pd


def _to_iso8601(value: Any) -> str | None:
    """Return ``value`` as an ISO-8601 string in UTC when possible."""

    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def _coerce_mapping(data: Any) -> dict[str, float]:
    if not isinstance(data, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in data.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


@dataclass(frozen=True)
class BaselineRecord:
    """Normalized representation of a baseline monitor run."""

    timestamp_range: tuple[str | None, str | None] = (None, None)
    live_time_s: float | None = None
    dilution_factor: float | None = None
    rates_Bq: dict[str, float] = field(default_factory=dict)
    rate_unc_Bq: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    po214_energy_stats: dict[str, float] | None = None

    @classmethod
    def from_summary_dict(cls, data: Mapping[str, Any]) -> "BaselineRecord":
        """Build a :class:`BaselineRecord` from ``summary['baseline']``."""

        timestamp_range = data.get("timestamp_range")
        if isinstance(timestamp_range, (list, tuple)) and len(timestamp_range) >= 2:
            start_iso = _to_iso8601(timestamp_range[0])
            end_iso = _to_iso8601(timestamp_range[1])
        else:
            start_iso = _to_iso8601(data.get("start"))
            end_iso = _to_iso8601(data.get("end"))

        live_time_val = data.get("live_time_s", data.get("live_time"))
        try:
            live_time = float(live_time_val) if live_time_val is not None else None
        except (TypeError, ValueError):
            live_time = None

        dilution_val = data.get("dilution_factor")
        try:
            dilution = float(dilution_val) if dilution_val is not None else None
        except (TypeError, ValueError):
            dilution = None

        rates = _coerce_mapping(data.get("rate_Bq"))
        rate_unc = _coerce_mapping(data.get("rate_unc_Bq"))

        counts_src = data.get("counts") or data.get("baseline_counts")
        counts = _coerce_mapping(counts_src)

        scales = _coerce_mapping(data.get("scales"))
        po214_stats = data.get("po214_energy_stats")
        if isinstance(po214_stats, Mapping):
            po214_stats = {
                key: float(value)
                for key, value in po214_stats.items()
                if isinstance(value, (int, float))
            }
        else:
            po214_stats = None

        return cls(
            timestamp_range=(start_iso, end_iso),
            live_time_s=live_time,
            dilution_factor=dilution,
            rates_Bq=rates,
            rate_unc_Bq=rate_unc,
            counts=counts,
            scales=scales,
            po214_energy_stats=po214_stats,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary representation."""

        data: dict[str, Any] = {
            "timestamp_range": [
                ts for ts in self.timestamp_range if ts is not None
            ],
            "live_time_s": self.live_time_s,
            "dilution_factor": self.dilution_factor,
            "rates_Bq": dict(self.rates_Bq),
            "rate_unc_Bq": dict(self.rate_unc_Bq),
            "counts": dict(self.counts),
        }
        if self.po214_energy_stats:
            data["po214_energy_stats"] = dict(self.po214_energy_stats)
        return data


def _efficiency_from_config(config: Mapping[str, Any], isotope: str) -> float | None:
    """Return the configured efficiency for ``isotope`` if finite."""

    tf_cfg = config.get("time_fit", {}) if isinstance(config, Mapping) else {}
    eff_cfg = tf_cfg.get(f"eff_{isotope.lower()}") if isinstance(tf_cfg, Mapping) else None
    if isinstance(eff_cfg, (list, tuple)):
        eff_cfg = eff_cfg[0] if eff_cfg else None
    if eff_cfg in (None, "null"):
        return 1.0
    try:
        eff = float(eff_cfg)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(eff) or eff <= 0:
        return None
    return eff


def get_fixed_background_for_time_fit(
    record: BaselineRecord,
    isotope: str,
    config: Mapping[str, Any],
) -> tuple[float, float, str] | None:
    """Return background inputs derived from ``record`` for a time fit."""

    rate = record.rates_Bq.get(isotope)
    unc = record.rate_unc_Bq.get(isotope)
    counts = record.counts.get(isotope)
    live_time = record.live_time_s

    eff = _efficiency_from_config(config, isotope)

    if rate is None and None not in (counts, live_time, eff):
        try:
            rate = float(counts) / (float(live_time) * float(eff))
        except (ZeroDivisionError, ValueError):
            rate = None
    if unc is None and None not in (counts, live_time, eff):
        try:
            unc = math.sqrt(abs(float(counts))) / (float(live_time) * float(eff))
        except (ZeroDivisionError, ValueError):
            unc = None

    if rate is None:
        return None

    if unc is None:
        unc = 0.0

    scale = record.scales.get(isotope)
    if scale is None and record.dilution_factor is not None and isotope in {"Po214", "Po218"}:
        scale = record.dilution_factor
    if scale is None:
        scale = 1.0

    scaled_rate = float(rate) * float(scale)
    scaled_unc = float(unc) * abs(float(scale))

    return scaled_rate, scaled_unc, "baseline_fixed"


def annotate_time_fit_provenance(
    entry: dict[str, Any],
    baseline_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach baseline provenance details to a time-fit summary entry."""

    record_data = baseline_details.get("record") if isinstance(baseline_details, Mapping) else None
    if not isinstance(record_data, Mapping):
        return entry
    record = BaselineRecord.from_summary_dict(record_data)

    rate = baseline_details.get("rate_Bq")
    if rate is None:
        return entry
    unc = baseline_details.get("unc_Bq")
    mode = baseline_details.get("mode", "baseline_fixed")

    entry = dict(entry)
    entry["background_source"] = mode
    entry["baseline_activity_Bq"] = float(rate)
    if unc is not None:
        entry["baseline_activity_unc_Bq"] = float(unc)
    ts_range = [ts for ts in record.timestamp_range if ts is not None]
    if ts_range:
        entry["baseline_source_range"] = ts_range
    if record.live_time_s is not None:
        entry["baseline_live_time_s"] = float(record.live_time_s)
    if record.dilution_factor is not None:
        entry["baseline_dilution_factor"] = float(record.dilution_factor)
    return entry


def check_baseline_drift(
    record: BaselineRecord,
    current_stats: Mapping[str, float] | None,
    *,
    energy_tol_MeV: float = 0.05,
    sigma_tol_fraction: float = 0.2,
) -> tuple[bool, str | None]:
    """Evaluate detector drift between baseline and assay runs."""

    base_stats = record.po214_energy_stats or {}
    cur_stats = current_stats or {}

    base_centroid = base_stats.get("centroid_MeV")
    cur_centroid = cur_stats.get("centroid_MeV")
    base_sigma = base_stats.get("sigma_MeV")
    cur_sigma = cur_stats.get("sigma_MeV")

    warnings: list[str] = []
    warn = False

    if None not in (base_centroid, cur_centroid):
        delta_e = abs(float(base_centroid) - float(cur_centroid))
        if delta_e > energy_tol_MeV:
            warn = True
            warnings.append(f"centroid shifted by {delta_e:.3f} MeV")

    if None not in (base_sigma, cur_sigma) and float(cur_sigma) > 0:
        ratio = abs(float(base_sigma) - float(cur_sigma)) / float(cur_sigma)
        if ratio > sigma_tol_fraction:
            warn = True
            warnings.append(f"sigma drift {ratio * 100:.1f}%")

    if not warn:
        return False, None

    message = "baseline spectral shape drifted relative to assay run"
    if warnings:
        message = f"{message} ({'; '.join(warnings)})"
    return True, message


__all__ = [
    "BaselineRecord",
    "get_fixed_background_for_time_fit",
    "annotate_time_fit_provenance",
    "check_baseline_drift",
]
