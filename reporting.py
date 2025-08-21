import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, List
import numpy as np


@dataclass
class Diagnostics:
    """Container for lightweight run diagnostics.

    Parameters
    ----------
    spectral_fit_fit_valid : bool | None
        Whether the spectral fit converged.
    time_fit_po214_fit_valid : bool | None
        Whether the Po214 time fit converged.
    time_fit_po218_fit_valid : bool | None
        Whether the Po218 time fit converged.
    n_events_loaded : int
        Total number of events loaded from file.
    n_events_discarded : int
        Number of events discarded by cuts.
    selected_analysis_modes : dict
        Summary of analysis mode selections.
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


def _extract_fit_valid(obj: Any, iso: str | None = None) -> bool | None:
    """Return fit validity flag from ``obj``."""
    if obj is None:
        return None
    params: Mapping[str, Any]
    if hasattr(obj, "params"):
        params = getattr(obj, "params")  # type: ignore[assignment]
    elif isinstance(obj, Mapping):
        params = obj
    else:
        return None
    if iso is not None:
        key = f"fit_valid_{iso}"
        if key in params:
            return bool(params.get(key))
    if "fit_valid" in params:
        return bool(params.get("fit_valid"))
    return None


def _binning_info(df_analysis, cfg: Mapping[str, Any]) -> tuple[str, float | None]:
    """Return (mode, width) for the spectral plot binning."""
    spec_cfg = cfg.get("spectral_fit", {})
    bin_cfg = spec_cfg.get("binning")
    if bin_cfg is not None:
        mode = str(bin_cfg.get("method", "adc")).lower()
    else:
        mode = str(spec_cfg.get("spectral_binning_mode", "adc")).lower()
    width: float | None = None
    if mode == "fd":
        energies = df_analysis.get("energy_MeV")
        if energies is not None:
            arr = np.asarray(energies)
            if arr.size > 0:
                q25, q75 = np.percentile(arr, [25, 75])
                iqr = q75 - q25
                if iqr > 0:
                    width = 2 * iqr / (arr.size ** (1 / 3))
    else:
        if bin_cfg is not None:
            width = bin_cfg.get("adc_bin_width", spec_cfg.get("adc_bin_width", 1))
        else:
            width = spec_cfg.get("adc_bin_width", 1)
    return mode, width


def build_diagnostics(summary, spectrum_results, time_fit_results, df_analysis, cfg) -> dict:
    """Construct a diagnostics dictionary for the current analysis run."""

    n_loaded = len(df_analysis)
    noise_removed = 0
    burst_removed = 0
    noise_cut = getattr(summary, "noise_cut", None)
    if isinstance(noise_cut, Mapping):
        noise_removed = int(noise_cut.get("removed_events", 0))
    burst_filter = getattr(summary, "burst_filter", None)
    if isinstance(burst_filter, Mapping):
        burst_removed = int(burst_filter.get("removed_events", 0))
    n_loaded += noise_removed + burst_removed
    n_discarded = n_loaded - len(df_analysis)

    mode, width = _binning_info(df_analysis, cfg)

    diag = {
        "spectral_fit_fit_valid": _extract_fit_valid(spectrum_results),
        "time_fit_po214_fit_valid": _extract_fit_valid(time_fit_results.get("Po214"), "Po214"),
        "time_fit_po218_fit_valid": _extract_fit_valid(time_fit_results.get("Po218"), "Po218"),
        "n_events_loaded": int(n_loaded),
        "n_events_discarded": int(n_discarded),
        "selected_analysis_modes": {
            "background_model": cfg.get("analysis", {}).get("background_model"),
            "spectral_unbinned_likelihood": bool(
                cfg.get("spectral_fit", {}).get("unbinned_likelihood", False)
            ),
            "binning_mode": mode,
            "binning_width": width,
        },
        "warnings": get_captured_warnings(),
    }

    return diag


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


__all__ = [
    "Diagnostics",
    "start_warning_capture",
    "get_captured_warnings",
    "build_diagnostics",
]
