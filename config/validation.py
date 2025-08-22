import logging
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
            "baseline range does not intersect analysis window: "
            f"[{start.isoformat()}, {end.isoformat()}] vs end {a_end_dt.isoformat()}"
        )

    if a_start_dt is not None and end <= a_start_dt:
        raise ValueError(
            "baseline range does not intersect analysis window: "
            f"[{start.isoformat()}, {end.isoformat()}] vs start {a_start_dt.isoformat()}"
        )

