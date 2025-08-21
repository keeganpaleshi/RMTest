from __future__ import annotations

from typing import Mapping
import logging

from utils import parse_time_arg


def validate_baseline_window(cfg: Mapping[str, object]) -> None:
    """Validate baseline window relative to analysis times.

    Checks
    -------
    a) baseline.start < analysis_end_time
    b) baseline duration > 0
    c) baseline is not entirely after the analysis window
    """
    analysis = cfg.get("analysis", {}) or {}
    baseline = cfg.get("baseline", {}) or {}
    rng = baseline.get("range")
    if not rng:
        return

    try:
        b_start = parse_time_arg(rng[0])
        b_end = parse_time_arg(rng[1])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid baseline.range {rng!r}: {exc}") from exc

    if b_end <= b_start:
        raise ValueError(
            f"Baseline window end {b_end.isoformat()} must be after start {b_start.isoformat()}"
        )

    a_end_raw = analysis.get("analysis_end_time")
    if a_end_raw is not None:
        a_end = parse_time_arg(a_end_raw)
        if b_start >= a_end:
            raise ValueError(
                f"Baseline start {b_start.isoformat()} occurs after analysis_end_time {a_end.isoformat()}"
            )

    a_start_raw = analysis.get("analysis_start_time")
    if a_start_raw is not None:
        a_start = parse_time_arg(a_start_raw)
        if b_start >= a_start and b_end <= a_start:
            raise ValueError(
                f"Baseline window {b_start.isoformat()}–{b_end.isoformat()} lies entirely after analysis window"
            )

    logging.debug(
        "Baseline window %s–%s validated against analysis window", b_start, b_end
    )
