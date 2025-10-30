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


def validate_radon_inference(cfg: dict) -> None:
    """Validate the radon inference configuration block."""

    radon_cfg = cfg.get("radon_inference")
    if not radon_cfg:
        return

    if not isinstance(radon_cfg, Mapping):
        raise ValueError("radon_inference must be a mapping of configuration values")

    isotopes = radon_cfg.get("source_isotopes")
    if not isinstance(isotopes, list) or not isotopes:
        raise ValueError(
            "radon_inference.source_isotopes must be a non-empty list of supported isotopes"
        )

    allowed_isotopes = {"Po214", "Po218"}
    invalid_isotopes = [iso for iso in isotopes if iso not in allowed_isotopes]
    if invalid_isotopes:
        raise ValueError(
            "radon_inference.source_isotopes contains unsupported entries: "
            + ", ".join(sorted(invalid_isotopes))
        )

    weights = radon_cfg.get("source_weights") or {}
    if not isinstance(weights, Mapping):
        raise ValueError("radon_inference.source_weights must be a mapping")
    extra_weights = [iso for iso in weights if iso not in isotopes]
    if extra_weights:
        raise ValueError(
            "radon_inference.source_weights has entries not listed in source_isotopes: "
            + ", ".join(sorted(extra_weights))
        )

    detection = radon_cfg.get("detection_efficiency")
    if not isinstance(detection, Mapping):
        raise ValueError(
            "radon_inference.detection_efficiency must be a mapping of isotopes to efficiencies"
        )

    missing_detection = [iso for iso in isotopes if detection.get(iso) is None]
    if missing_detection:
        suffix = ", ".join(sorted(missing_detection))
        prefix = "radon_inference.detection_efficiency must provide values for each source isotope"
        if radon_cfg.get("enabled"):
            prefix += " when radon_inference.enabled is true"
        raise ValueError(f"{prefix} (missing: {suffix})")

    transport = radon_cfg.get("transport_efficiency")
    if transport is not None and not (0 < transport <= 1.5):
        raise ValueError(
            "radon_inference.transport_efficiency must lie within the interval (0, 1.5]"
        )

    retention = radon_cfg.get("retention_efficiency")
    if retention is not None and not (0 < retention <= 1.5):
        raise ValueError(
            "radon_inference.retention_efficiency must lie within the interval (0, 1.5]"
        )

    external = radon_cfg.get("external_rn")
    if external is not None:
        if not isinstance(external, Mapping):
            raise ValueError("radon_inference.external_rn must be a mapping")

        mode = external.get("mode")
        if mode == "constant":
            constant = external.get("constant_bq_per_m3")
            if constant is None or constant <= 0:
                raise ValueError(
                    "radon_inference.external_rn.constant_bq_per_m3 must be > 0 when mode is 'constant'"
                )
        elif mode == "file":
            file_path = external.get("file_path")
            if not file_path:
                raise ValueError(
                    "radon_inference.external_rn.file_path must be provided when mode is 'file'"
                )
