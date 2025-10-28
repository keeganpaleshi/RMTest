"""Utilities for handling monitor-only baseline records.

The functions in this module provide a stable representation of the
``summary["baseline"]`` block written by :mod:`analyze`.  They also expose
helpers for re-using those baseline measurements when fitting time-series
decays with a fixed background component.

The long-lived, monitor-only baseline run is represented by
``BaselineRecord``.  It stores the intrinsic (monitor-only) Po-214 activity
and uncertainty, the live time of that run, the dilution factor required to
map the monitor-only activity into an assay with additional plumbing, and the
timestamp range that was integrated.  Optional calibration metadata is also
kept so that later assays can sanity-check detector stability.

Typical usage::

    record = ingest_baseline_record(summary["baseline"])
    fixed = get_fixed_background_for_time_fit(record, "Po214", cfg)
    annotate_time_fit_with_baseline(summary, record, {"Po214": fixed})

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Sequence

from baseline_utils import compute_dilution_factor


@dataclass
class BaselineRecord:
    """Structured representation of a monitor-only baseline measurement."""

    schema_version: int = 1
    live_time_s: float | None = None
    time_range: tuple[str | None, str | None] = (None, None)
    rates_Bq: dict[str, float] = field(default_factory=dict)
    rate_unc_Bq: dict[str, float] = field(default_factory=dict)
    dilution_factor: float | None = None
    calibration_snapshot: dict[str, Any] = field(default_factory=dict)

    def get_rate(self, isotope: str) -> float | None:
        return self.rates_Bq.get(isotope)

    def get_uncertainty(self, isotope: str) -> float | None:
        return self.rate_unc_Bq.get(isotope)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the record."""

        return {
            "schema_version": self.schema_version,
            "live_time_s": self.live_time_s,
            "time_range": list(self.time_range),
            "rates_Bq": dict(self.rates_Bq),
            "rate_unc_Bq": dict(self.rate_unc_Bq),
            "dilution_factor": self.dilution_factor,
            "calibration_snapshot": dict(self.calibration_snapshot),
        }


def _to_iso(value: Any) -> str | None:
    """Return an ISO-8601 string for ``value`` if possible."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            if value.tzinfo.utcoffset(value) == timezone.utc.utcoffset(value):
                return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return value.isoformat()
        return value.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value)


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` as ``float`` when possible, otherwise ``None``."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_time_range(data: Mapping[str, Any]) -> tuple[str | None, str | None]:
    time_range = data.get("time_range")
    if isinstance(time_range, Sequence) and len(time_range) >= 2:
        start, end = time_range[0], time_range[1]
    else:
        start = data.get("start")
        end = data.get("end")
    return (_to_iso(start), _to_iso(end))


def ingest_baseline_record(baseline_section: Mapping[str, Any] | None) -> BaselineRecord | None:
    """Return :class:`BaselineRecord` parsed from ``summary['baseline']``."""

    if not baseline_section:
        return None

    record_src = baseline_section.get("record") if isinstance(baseline_section, Mapping) else None
    if isinstance(record_src, Mapping):
        data = dict(record_src)
        data.setdefault("rates_Bq", record_src.get("rates_Bq", record_src.get("rate_Bq", {})))
        data.setdefault("rate_unc_Bq", record_src.get("rate_unc_Bq", baseline_section.get("rate_unc_Bq", {})))
        data.setdefault("dilution_factor", record_src.get("dilution_factor", baseline_section.get("dilution_factor")))
        data.setdefault("live_time_s", record_src.get("live_time_s", baseline_section.get("live_time")))
        data.setdefault("calibration_snapshot", record_src.get("calibration_snapshot", baseline_section.get("calibration_snapshot", {})))
    else:
        data = dict(baseline_section)
        data.setdefault("rates_Bq", baseline_section.get("rate_Bq", {}))
        data.setdefault("rate_unc_Bq", baseline_section.get("rate_unc_Bq", {}))
        data.setdefault("live_time_s", baseline_section.get("live_time"))
        data.setdefault("calibration_snapshot", baseline_section.get("calibration_snapshot", {}))

    time_range = _normalise_time_range(data)
    live_time = _coerce_float(data.get("live_time_s"))

    rates_raw = data.get("rates_Bq", {})
    rate_unc_raw = data.get("rate_unc_Bq", {})
    rates: dict[str, float] = {}
    rate_unc: dict[str, float] = {}
    if isinstance(rates_raw, Mapping):
        for iso, val in rates_raw.items():
            coerced = _coerce_float(val)
            if coerced is not None:
                rates[str(iso)] = coerced
    if isinstance(rate_unc_raw, Mapping):
        for iso, val in rate_unc_raw.items():
            coerced = _coerce_float(val)
            if coerced is not None:
                rate_unc[str(iso)] = coerced

    dilution = _coerce_float(data.get("dilution_factor"))
    calibration_snapshot = data.get("calibration_snapshot", {})
    if not isinstance(calibration_snapshot, Mapping):
        calibration_snapshot = {}

    return BaselineRecord(
        live_time_s=live_time,
        time_range=time_range,
        rates_Bq=rates,
        rate_unc_Bq=rate_unc,
        dilution_factor=dilution,
        calibration_snapshot=dict(calibration_snapshot),
    )


def _analysis_scale(config: Mapping[str, Any], record: BaselineRecord) -> float:
    baseline_cfg = config.get("baseline", {}) if isinstance(config, Mapping) else {}
    monitor = baseline_cfg.get("monitor_volume_l")
    sample = baseline_cfg.get("sample_volume_l")
    try:
        return compute_dilution_factor(monitor, sample)
    except Exception:
        if record.dilution_factor is not None:
            return float(record.dilution_factor)
        return 1.0


def get_fixed_background_for_time_fit(
    baseline_record: BaselineRecord | Mapping[str, Any] | None,
    isotope: str,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return scaled background information for ``isotope``.

    The returned dictionary contains the intrinsic (monitor-only) rate and
    uncertainty together with the scaled values appropriate for the assay run.
    It is suitable for feeding into :func:`time_fitting.two_pass_time_fit`.
    """

    if baseline_record is None:
        return None
    if not isinstance(baseline_record, BaselineRecord):
        baseline_record = ingest_baseline_record(baseline_record)
        if baseline_record is None:
            return None

    intrinsic = baseline_record.get_rate(isotope)
    if intrinsic is None:
        return None
    intrinsic_unc = baseline_record.get_uncertainty(isotope) or 0.0
    scale = _analysis_scale(config, baseline_record)
    background = intrinsic * scale
    background_unc = intrinsic_unc * abs(scale)
    return {
        "mode": "baseline_fixed",
        "intrinsic_rate_Bq": intrinsic,
        "intrinsic_unc_Bq": intrinsic_unc,
        "background_rate_Bq": background,
        "background_unc_Bq": background_unc,
        "scale_factor": scale,
        "time_range": list(baseline_record.time_range),
    }


def annotate_time_fit_with_baseline(
    summary: MutableMapping[str, Any] | Any,
    baseline_record: BaselineRecord | Mapping[str, Any] | None,
    provenance: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Attach baseline provenance to ``summary['time_fit']`` entries."""

    if not provenance:
        return
    if baseline_record is None:
        return
    if not isinstance(baseline_record, BaselineRecord):
        baseline_record = ingest_baseline_record(baseline_record)
        if baseline_record is None:
            return

    time_fit = getattr(summary, "time_fit", None)
    if time_fit is None and isinstance(summary, MutableMapping):
        time_fit = summary.get("time_fit")
    if time_fit is None:
        return

    for iso, info in provenance.items():
        if not info:
            continue
        entry = dict(time_fit.get(iso, {}))
        entry["background_source"] = info.get("mode", "baseline_fixed")
        entry["baseline_source_range"] = list(baseline_record.time_range)
        entry["baseline_activity_Bq"] = info.get("background_rate_Bq")
        entry["baseline_activity_unc_Bq"] = info.get("background_unc_Bq")
        entry["baseline_intrinsic_activity_Bq"] = info.get("intrinsic_rate_Bq")
        entry["baseline_scale_factor"] = info.get("scale_factor")
        time_fit[iso] = entry

    baseline_section = getattr(summary, "baseline", None)
    if isinstance(summary, MutableMapping):
        baseline_section = summary.get("baseline", baseline_section)
    if isinstance(baseline_section, MutableMapping):
        baseline_section.setdefault("record", baseline_record.as_dict())


def evaluate_baseline_drift(
    baseline_record: BaselineRecord | Mapping[str, Any] | None,
    calibration_summary: Mapping[str, Any] | None,
    *,
    energy_tolerance_mev: float = 0.05,
    sigma_relative_tolerance: float = 0.2,
) -> tuple[bool, str | None]:
    """Compare baseline spectral calibration with the current run."""

    if baseline_record is None or calibration_summary is None:
        return False, None
    if not isinstance(baseline_record, BaselineRecord):
        baseline_record = ingest_baseline_record(baseline_record)
        if baseline_record is None:
            return False, None

    cal_snapshot = baseline_record.calibration_snapshot or {}
    base_energy = _coerce_float(cal_snapshot.get("po214_centroid_mev"))
    base_sigma = _coerce_float(cal_snapshot.get("sigma_E"))

    peaks = calibration_summary.get("peaks") if isinstance(calibration_summary, Mapping) else None
    current_energy = None
    if isinstance(peaks, Mapping):
        current_energy = _coerce_float(peaks.get("Po214", {}).get("centroid_mev"))
    current_sigma = _coerce_float(calibration_summary.get("sigma_E"))

    if base_energy is None or current_energy is None:
        return False, None

    delta_energy = abs(current_energy - base_energy)
    sigma_warning = False
    ratio = None
    if base_sigma and current_sigma:
        try:
            ratio = float(current_sigma) / float(base_sigma)
            sigma_warning = abs(ratio - 1.0) > sigma_relative_tolerance
        except ZeroDivisionError:
            sigma_warning = False

    energy_warning = delta_energy > energy_tolerance_mev
    warning = energy_warning or sigma_warning
    if not warning:
        return False, None

    details = []
    details.append(f"ΔE={delta_energy:.3f} MeV")
    if ratio is not None:
        details.append(f"σ_ratio={ratio:.2f}")
    message = "baseline spectral shape drifted relative to assay run"
    if details:
        message += f" ({', '.join(details)})"
    return True, message


def apply_baseline_drift_warning(
    summary: MutableMapping[str, Any] | Any,
    warning_flag: bool,
    message: str | None,
) -> None:
    """Record baseline drift information inside ``summary['diagnostics']``."""

    diagnostics = getattr(summary, "diagnostics", None)
    if diagnostics is None and isinstance(summary, MutableMapping):
        diagnostics = summary.get("diagnostics")
    if diagnostics is None:
        diagnostics = {}
        if isinstance(summary, MutableMapping):
            summary["diagnostics"] = diagnostics
        else:
            setattr(summary, "diagnostics", diagnostics)

    diagnostics["baseline_compat_warning"] = bool(warning_flag)
    if warning_flag and message:
        warnings_list = diagnostics.setdefault("warnings", [])
        if message not in warnings_list:
            warnings_list.append(message)


__all__ = [
    "BaselineRecord",
    "ingest_baseline_record",
    "get_fixed_background_for_time_fit",
    "annotate_time_fit_with_baseline",
    "evaluate_baseline_drift",
    "apply_baseline_drift_warning",
]
