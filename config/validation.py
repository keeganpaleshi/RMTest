import logging
from collections.abc import Mapping
from datetime import datetime
from utils import to_utc_datetime


def validate_baseline_window(cfg: dict) -> None:
    """Validate baseline window relative to analysis range.

    Parameters
    ----------
    cfg : dict
        Configuration mapping that may contain ``baseline`` and ``analysis``
        sections.  ``baseline.range`` should be a two element iterable of
        timestamps.  ``analysis.analysis_end_time`` and
        ``analysis.analysis_start_time`` may be ``None``.

    Raises
    ------
    ValueError
        If the baseline interval is malformed or does not overlap with the
        analysis window.
    """
    baseline = cfg.get("baseline", {})
    b_range = baseline.get("range")
    if not b_range or len(b_range) != 2:
        return

    try:
        start = to_utc_datetime(b_range[0])
        end = to_utc_datetime(b_range[1])
    except Exception as exc:  # pragma: no cover - safety net
        raise ValueError(f"Invalid baseline range {b_range!r} -> {exc}") from exc

    if end <= start:
        raise ValueError(
            f"Baseline end {end.isoformat()} must be after start {start.isoformat()}"
        )

    analysis = cfg.get("analysis", {})
    a_end = analysis.get("analysis_end_time")
    a_start = analysis.get("analysis_start_time")
    a_end_dt = to_utc_datetime(a_end) if a_end is not None else None
    a_start_dt = to_utc_datetime(a_start) if a_start is not None else None

    if a_end_dt is not None and start >= a_end_dt:
        raise ValueError(
            "baseline.start >= analysis_end_time: "
            f"{start.isoformat()} vs {a_end_dt.isoformat()}. "
            "Check baseline.range and analysis.analysis_end_time"
        )

    if a_start_dt is not None and end <= a_start_dt:
        raise ValueError(
            "Baseline interval entirely before analysis window: "
            f"[{start.isoformat()}, {end.isoformat()}] vs "
            f"start {a_start_dt.isoformat()}"
        )


def _validate_efficiency_value(name: str, value) -> float:
    """Ensure efficiencies are numeric and within allowed range."""

    try:
        eff = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a numeric value") from exc
    if eff <= 0.0 or eff > 1.5:
        raise ValueError(f"{name} must be in the range (0, 1.5]")
    return eff


def validate_radon_inference(cfg: dict) -> None:
    """Validate radon inference configuration options."""

    radon_cfg = cfg.get("radon_inference")
    if not radon_cfg:
        return

    if not isinstance(radon_cfg, Mapping):  # pragma: no cover - schema guard
        raise ValueError("radon_inference must be a mapping of configuration keys")

    enabled = bool(radon_cfg.get("enabled", False))
    isotopes = radon_cfg.get("source_isotopes")
    if not isotopes or not isinstance(isotopes, list):
        raise ValueError(
            "radon_inference.source_isotopes must be a non-empty list of isotopes"
        )

    allowed_isotopes = {"Po214", "Po218"}
    invalid_isotopes = [iso for iso in isotopes if iso not in allowed_isotopes]
    if invalid_isotopes:
        raise ValueError(
            "radon_inference.source_isotopes contains unsupported entries: "
            + ", ".join(sorted(invalid_isotopes))
        )

    isotopes_set = set(isotopes)

    weights = radon_cfg.get("source_weights") or {}
    if not isinstance(weights, Mapping):
        raise ValueError("radon_inference.source_weights must be a mapping of weights")
    extra_weights = set(weights) - isotopes_set
    if extra_weights:
        raise ValueError(
            "radon_inference.source_weights includes isotopes not in source_isotopes: "
            + ", ".join(sorted(extra_weights))
        )

    detection = radon_cfg.get("detection_efficiency") or {}
    if not isinstance(detection, Mapping):
        raise ValueError(
            "radon_inference.detection_efficiency must be a mapping of efficiencies"
        )
    if enabled:
        missing_detection = [iso for iso in isotopes if iso not in detection]
        if missing_detection:
            raise ValueError(
                "radon_inference.detection_efficiency missing entries for: "
                + ", ".join(sorted(missing_detection))
            )

    transport_eff = radon_cfg.get("transport_efficiency")
    if transport_eff is not None:
        _validate_efficiency_value("radon_inference.transport_efficiency", transport_eff)

    retention_eff = radon_cfg.get("retention_efficiency")
    if retention_eff is not None:
        _validate_efficiency_value("radon_inference.retention_efficiency", retention_eff)

    external = radon_cfg.get("external_rn") or {}
    if not isinstance(external, Mapping):
        raise ValueError("radon_inference.external_rn must be a mapping")

    mode = external.get("mode")
    if mode == "constant":
        value = external.get("constant_bq_per_m3")
        if value is None:
            raise ValueError(
                "radon_inference.external_rn.constant_bq_per_m3 must be provided when mode is 'constant'"
            )
        try:
            numeric_val = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "radon_inference.external_rn.constant_bq_per_m3 must be numeric"
            ) from exc
        if numeric_val <= 0:
            raise ValueError(
                "radon_inference.external_rn.constant_bq_per_m3 must be greater than 0"
            )
    elif mode == "file":
        file_path = external.get("file_path")
        if not file_path or not str(file_path).strip():
            raise ValueError(
                "radon_inference.external_rn.file_path must be set when mode is 'file'"
            )
