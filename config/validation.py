from __future__ import annotations

from typing import Mapping, Any

from utils.time_utils import parse_timestamp


def validate_baseline_window(cfg: Mapping[str, Any]) -> None:
    """Validate baseline range against analysis window.

    Raises ValueError with a helpful message when the baseline window is
    malformed or occurs entirely after the analysis interval.
    """
    baseline = cfg.get("baseline", {}) if isinstance(cfg, Mapping) else {}
    rng = baseline.get("range")
    if not rng or len(rng) < 2:
        return

    start = parse_timestamp(rng[0])
    end = parse_timestamp(rng[1])
    if end <= start:
        raise ValueError(
            f"Baseline end {end} must be after start {start}. Check baseline.range."
        )

    analysis = cfg.get("analysis", {}) if isinstance(cfg, Mapping) else {}
    a_end_raw = analysis.get("analysis_end_time")
    if a_end_raw is not None:
        a_end = parse_timestamp(a_end_raw)
        if start >= a_end:
            raise ValueError(
                "Baseline window "
                f"[{start}, {end}] occurs after analysis end {a_end}. "
                "Adjust baseline.range or analysis.analysis_end_time."
            )
