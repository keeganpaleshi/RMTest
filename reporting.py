import logging
from dataclasses import dataclass, field, asdict
from typing import List, Mapping, Any


@dataclass
class Diagnostics:
    """Container for lightweight run diagnostics."""

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


def default_diagnostics() -> dict:
    """Return a diagnostics dictionary with default values."""
    return asdict(Diagnostics())


def _fit_valid(fit: Any) -> bool | None:
    """Extract the ``fit_valid`` flag from a fit result object."""
    try:
        from fitting import FitResult  # lazy import to avoid cycle
    except Exception:  # pragma: no cover - fallback if import fails
        FitResult = object  # type: ignore

    if fit is None:
        return None
    if isinstance(fit, dict):
        return bool(fit.get("fit_valid"))
    if isinstance(fit, FitResult):
        return bool(fit.params.get("fit_valid"))
    return None


def build_diagnostics(summary: Mapping[str, Any], spectrum_results: Any, time_fit_results: Mapping[str, Any], df_analysis, cfg: Mapping[str, Any]) -> dict:
    """Construct diagnostics information for the current run."""

    diag = default_diagnostics()

    diag["spectral_fit_fit_valid"] = _fit_valid(spectrum_results)
    diag["time_fit_po214_fit_valid"] = _fit_valid(time_fit_results.get("Po214"))
    diag["time_fit_po218_fit_valid"] = _fit_valid(time_fit_results.get("Po218"))

    n_removed_noise = summary.get("noise_cut", {}).get("removed_events", 0)
    n_removed_burst = summary.get("burst_filter", {}).get("removed_events", 0)
    discarded = int(n_removed_noise) + int(n_removed_burst)
    diag["n_events_discarded"] = discarded
    diag["n_events_loaded"] = discarded + len(df_analysis)

    spec_cfg = cfg.get("spectral_fit", {})
    bin_cfg = spec_cfg.get("binning", {}) if isinstance(spec_cfg.get("binning"), Mapping) else {}
    bin_mode = (bin_cfg.get("method") or spec_cfg.get("spectral_binning_mode") or "adc")
    if bin_mode == "fd":
        bin_width = bin_cfg.get("default_bins") or spec_cfg.get("fd_hist_bins")
    else:
        bin_width = bin_cfg.get("adc_bin_width") or spec_cfg.get("adc_bin_width")
    diag["selected_analysis_modes"] = {
        "background_model": spec_cfg.get("background_model"),
        "spectral_fit_unbinned_likelihood": spec_cfg.get("unbinned_likelihood"),
        "spectrum_binning_mode": bin_mode,
        "spectrum_binning_width": bin_width,
    }

    diag["warnings"] = get_captured_warnings()
    return diag


__all__ = [
    "Diagnostics",
    "start_warning_capture",
    "get_captured_warnings",
    "default_diagnostics",
    "build_diagnostics",
]
