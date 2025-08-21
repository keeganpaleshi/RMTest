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
    time_fit_po218_fit_valid: bool | None = None
    n_events_loaded: int = 0
    n_events_discarded: int = 0
    selected_analysis_modes: dict = field(default_factory=dict)
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


def build_diagnostics(summary, spectrum_results, time_fit_results, df_analysis, cfg) -> dict:
    """Construct a diagnostics dictionary for the analysis run."""

    def _get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    def _fit_valid(res):
        if hasattr(res, "params"):
            return bool(res.params.get("fit_valid", False))
        if isinstance(res, dict):
            return bool(res.get("fit_valid", False))
        return False

    noise_removed = _get(_get(summary, "noise_cut", {}), "removed_events", 0) or 0
    burst_removed = _get(_get(summary, "burst_filter", {}), "removed_events", 0) or 0
    n_loaded = int(len(df_analysis) + noise_removed + burst_removed)
    diagnostics = {
        "spectral_fit_fit_valid": _fit_valid(spectrum_results),
        "time_fit_po214_fit_valid": _fit_valid(_get(time_fit_results, "Po214")),
        "n_events_loaded": n_loaded,
        "n_events_discarded": int(n_loaded - len(df_analysis)),
        "selected_analysis_modes": {
            "background_model": _get(cfg.get("analysis", {}), "background_model"),
            "unbinned_likelihood": bool(
                cfg.get("spectral_fit", {}).get("unbinned_likelihood", False)
            ),
            "spectrum_binning": {},
        },
        "warnings": get_captured_warnings(),
    }

    if _get(time_fit_results, "Po218") is not None:
        diagnostics["time_fit_po218_fit_valid"] = _fit_valid(
            _get(time_fit_results, "Po218")
        )

    bin_cfg = cfg.get("spectral_fit", {}).get("binning")
    if bin_cfg is not None:
        mode = str(bin_cfg.get("method", "adc")).lower()
        width = (
            bin_cfg.get("adc_bin_width")
            if mode == "adc"
            else bin_cfg.get("default_bins")
        )
    else:
        mode = str(cfg.get("spectral_fit", {}).get("spectral_binning_mode", "adc")).lower()
        width = (
            cfg.get("spectral_fit", {}).get("adc_bin_width")
            if mode == "adc"
            else cfg.get("spectral_fit", {}).get("fd_hist_bins")
        )
    diagnostics["selected_analysis_modes"]["spectrum_binning"] = {
        "mode": mode,
        "width": width,
    }

    return diagnostics


__all__ = ["Diagnostics", "start_warning_capture", "get_captured_warnings", "build_diagnostics"]
