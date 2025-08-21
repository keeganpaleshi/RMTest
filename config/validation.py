"""Configuration validation utilities."""
from __future__ import annotations

from datetime import datetime
import logging

from utils import to_utc_datetime

__all__ = ["validate_baseline_window"]

logger = logging.getLogger(__name__)


def validate_baseline_window(cfg: dict) -> None:
    """Validate baseline window against analysis times.

    Checks that the baseline start precedes the analysis end time,
    the baseline duration is positive, and the baseline window is not
    completely outside the analysis interval.

    Parameters
    ----------
    cfg : dict
        Loaded configuration mapping.

    Raises
    ------
    ValueError
        If the baseline configuration is inconsistent.
    """
    baseline_cfg = cfg.get("baseline", {})
    analysis_cfg = cfg.get("analysis", {})

    bl_range = baseline_cfg.get("range")
    if not bl_range or len(bl_range) != 2:
        return

    try:
        bl_start = to_utc_datetime(bl_range[0])
        bl_end = to_utc_datetime(bl_range[1])
    except Exception as exc:  # pragma: no cover - malformed times handled elsewhere
        raise ValueError(f"Invalid baseline.range: {bl_range!r}") from exc

    if bl_end <= bl_start:
        raise ValueError(
            "baseline end time must be after start time: "
            f"start={bl_start.isoformat()} end={bl_end.isoformat()}"
        )

    analysis_end = analysis_cfg.get("analysis_end_time")
    analysis_start = analysis_cfg.get("analysis_start_time")
    if analysis_end is not None:
        analysis_end_dt = to_utc_datetime(analysis_end)
        if bl_start >= analysis_end_dt:
            raise ValueError(
                "baseline start is after analysis end: "
                f"baseline_start={bl_start.isoformat()} analysis_end={analysis_end_dt.isoformat()}"
            )
    if analysis_start is not None:
        analysis_start_dt = to_utc_datetime(analysis_start)
        if bl_end <= analysis_start_dt:
            raise ValueError(
                "baseline window lies entirely before analysis start: "
                f"baseline_end={bl_end.isoformat()} analysis_start={analysis_start_dt.isoformat()}"
            )

