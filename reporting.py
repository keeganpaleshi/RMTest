"""Utilities for collecting diagnostics information during analysis."""

from dataclasses import dataclass, field
import logging
from typing import Mapping, Any


@dataclass
class Diagnostics:
    """Minimal diagnostics written to ``summary.json``.

    Attributes
    ----------
    spectral_fit_fit_valid : bool
        ``True`` when the spectral fit converged with a valid covariance.
    time_fit_po214_fit_valid : bool
        Validity flag for the Po214 time-series fit.
    n_events_loaded : int
        Number of events loaded from input files.
    n_events_discarded : int
        Number of events discarded by cuts and filters.
    warnings : list[str]
        Collected warning messages emitted during processing.
    """

    spectral_fit_fit_valid: bool = False
    time_fit_po214_fit_valid: bool = False
    n_events_loaded: int = 0
    n_events_discarded: int = 0
    warnings: list[str] = field(default_factory=list)


class WarningCollector(logging.Handler):
    """Logging handler that captures warning-level log messages."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(level=logging.WARNING)
        self.warnings: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        self.warnings.append(record.getMessage())


def build_diagnostics(
    summary: Mapping[str, Any],
    *,
    n_events_loaded: int = 0,
    n_events_discarded: int = 0,
    warning_collector: "WarningCollector | None" = None,
) -> Diagnostics:
    """Create a :class:`Diagnostics` instance from summary information."""

    spectral_ok = bool(summary.get("spectral_fit", {}).get("fit_valid", False))
    time_ok = bool(
        summary.get("time_fit", {}).get("Po214", {}).get("fit_valid", False)
    )
    warnings = list(warning_collector.warnings) if warning_collector else []
    return Diagnostics(
        spectral_fit_fit_valid=spectral_ok,
        time_fit_po214_fit_valid=time_ok,
        n_events_loaded=int(n_events_loaded),
        n_events_discarded=int(n_events_discarded),
        warnings=warnings,
    )
