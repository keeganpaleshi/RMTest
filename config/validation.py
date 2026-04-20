import logging
from collections.abc import Mapping
from datetime import datetime
from utils import to_utc_datetime


def validate_baseline_window(cfg: dict) -> None:
    """Validate the baseline window format.

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
        If the baseline interval is malformed.
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

    # The baseline may come from a separate run or from a quiescent period
    # outside the active assay window, so no overlap with the analysis window
    # is required here.


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


def validate_lucas_bridge(cfg: dict) -> None:
    """Validate the ``lucas_bridge`` configuration section.

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration.

    Raises
    ------
    ValueError
        If any lucas_bridge settings are invalid.
    """
    bridge = cfg.get("lucas_bridge")
    if not bridge or not isinstance(bridge, Mapping):
        return
    if not bridge.get("enabled", False):
        return

    # Validate comparison_target
    valid_targets = {
        "baseline_corrected_combined",
        "radon_activity",
        "radon_combined",
        "po214",
        "po218",
    }
    target = bridge.get("comparison_target", "baseline_corrected_combined")
    if target not in valid_targets:
        raise ValueError(
            f"lucas_bridge.comparison_target must be one of {valid_targets}, "
            f"got {target!r}"
        )

    # Validate volume_convention
    valid_volumes = {"monitor", "sample", "total"}
    vol_conv = bridge.get("volume_convention", "monitor")
    if vol_conv not in valid_volumes:
        raise ValueError(
            f"lucas_bridge.volume_convention must be one of {valid_volumes}, "
            f"got {vol_conv!r}"
        )

    # Validate assay_files exist
    assay_files = bridge.get("assay_files", [])
    if not assay_files:
        logger = logging.getLogger(__name__)
        logger.warning("lucas_bridge is enabled but no assay_files specified")

    # Validate date_range if provided
    selection = bridge.get("selection") or {}
    date_range = selection.get("date_range")
    if date_range is not None:
        if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
            raise ValueError(
                "lucas_bridge.selection.date_range must be [start_iso, end_iso]"
            )
        try:
            datetime.fromisoformat(str(date_range[0]))
            datetime.fromisoformat(str(date_range[1]))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"lucas_bridge.selection.date_range contains invalid ISO dates: {exc}"
            ) from exc
