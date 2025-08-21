"""Utility functions for writing summary diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import logging
from typing import Iterable, List, Dict, Any

from io_utils import write_summary as _write_summary


@dataclass
class Diagnostics:
    """Diagnostics block to be embedded in ``summary.json``."""

    spectral_fit_fit_valid: bool = False
    time_fit_po214_fit_valid: bool = False
    n_events_loaded: int = 0
    n_events_discarded: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@contextmanager
def capture_warnings() -> Iterable[List[str]]:
    """Capture ``logging.WARNING`` level messages.

    Examples
    --------
    >>> with capture_warnings() as w:
    ...     logging.getLogger(__name__).warning("problem")
    >>> w
    ['problem']
    """

    records: List[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
            if record.levelno >= logging.WARNING:
                records.append(record.getMessage())

    handler = _ListHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield records
    finally:
        root.removeHandler(handler)


def write_summary(summary: Dict[str, Any], output_dir: str, diagnostics: Diagnostics | Dict[str, Any], **kwargs) -> Any:
    """Write a ``summary.json`` file including a diagnostics block.

    Parameters
    ----------
    summary:
        Base summary information.
    output_dir:
        Directory where ``summary.json`` should be created.
    diagnostics:
        Diagnostics information either as :class:`Diagnostics` or ``dict``.
    **kwargs:
        Additional keyword arguments passed through to
        :func:`io_utils.write_summary`.
    """

    if isinstance(diagnostics, Diagnostics):
        diag_dict = diagnostics.to_dict()
    else:
        diag_dict = dict(diagnostics)

    summary = dict(summary)  # copy
    summary["diagnostics"] = diag_dict
    return _write_summary(output_dir, summary, **kwargs)
