import logging
from dataclasses import dataclass, field
from typing import List


@dataclass
class Diagnostics:
    """Container for lightweight run diagnostics.

    Parameters
    ----------
    spectral_fit_fit_valid : bool | None
        Whether the spectral fit converged.
    time_fit_po214_fit_valid : bool | None
        Whether the Po214 time fit converged.
    n_events_loaded : int
        Total number of events loaded from file.
    n_events_discarded : int
        Number of events discarded by cuts.
    warnings : list[str]
        Collected log warning messages.
    """

    spectral_fit_fit_valid: bool | None = None
    time_fit_po214_fit_valid: bool | None = None
    n_events_loaded: int = 0
    n_events_discarded: int = 0
    warnings: List[str] = field(default_factory=list)


class _WarningCapture(logging.Handler):
    """Logging handler that stores warning messages."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple
        self.messages.append(record.getMessage())


_warning_handler: _WarningCapture | None = None


def start_warning_capture() -> None:
    """Begin collecting warning messages from the root logger."""
    global _warning_handler
    if _warning_handler is None:
        _warning_handler = _WarningCapture()
        logging.getLogger().addHandler(_warning_handler)


def get_captured_warnings() -> List[str]:
    """Return and clear captured warning messages."""
    global _warning_handler
    if _warning_handler is None:
        return []
    logging.getLogger().removeHandler(_warning_handler)
    msgs = list(_warning_handler.messages)
    _warning_handler = None
    return msgs


__all__ = ["Diagnostics", "start_warning_capture", "get_captured_warnings"]
