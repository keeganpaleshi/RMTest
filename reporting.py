"""Helpers for generating reporting information such as diagnostics blocks."""
from __future__ import annotations

from typing import Any
import logging


def diagnostics_block(**kwargs: Any) -> dict[str, Any]:
    """Return a diagnostics block with standard fields.

    Parameters are optional; missing fields are filled with defaults. Unknown
    keys are ignored to keep the interface forward compatible.
    """
    return {
        "spectral_fit_fit_valid": bool(kwargs.get("spectral_fit_fit_valid", False)),
        "time_fit_po214_fit_valid": bool(
            kwargs.get("time_fit_po214_fit_valid", False)
        ),
        "n_events_loaded": int(kwargs.get("n_events_loaded", 0)),
        "n_events_discarded": int(kwargs.get("n_events_discarded", 0)),
        "warnings": list(kwargs.get("warnings", [])),
    }


class WarningCaptureHandler(logging.Handler):
    """Logging handler that collects warning messages."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple
        if record.levelno >= logging.WARNING:
            self.messages.append(record.getMessage())
