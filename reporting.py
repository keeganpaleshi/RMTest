import logging
from copy import deepcopy
from typing import Any, Dict, List, Mapping


DEFAULT_DIAGNOSTICS: Dict[str, Any] = {
    "spectral_fit_fit_valid": None,
    "time_fit_po214_fit_valid": None,
    "time_fit_po218_fit_valid": None,
    "n_events_loaded": 0,
    "n_events_discarded": 0,
    "selected_analysis_modes": {},
    "warnings": [],
    "baseline_quality": None,
}


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


def build_diagnostics(
    summary: Mapping[str, Any],
    spectrum_results: Mapping[str, Any] | Any,
    time_fit_results: Mapping[str, Any],
    df_analysis,
    cfg: Mapping[str, Any],
    events_df=None,
    calibration: Mapping[str, Any] | None = None,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """Return a diagnostics dictionary for the run."""

    diagnostics = deepcopy(DEFAULT_DIAGNOSTICS)

    # Spectral fit validity
    fit_valid = None
    if spectrum_results is not None:
        if isinstance(spectrum_results, Mapping):
            fit_valid = spectrum_results.get("fit_valid")
        else:
            fit_valid = getattr(getattr(spectrum_results, "params", {}), "get", lambda *a: None)(
                "fit_valid"
            )
    diagnostics["spectral_fit_fit_valid"] = (
        bool(fit_valid) if fit_valid is not None else None
    )

    # Time-fit validity for Po214 and Po218
    tf214 = time_fit_results.get("Po214") if time_fit_results else None
    if tf214 is not None:
        if isinstance(tf214, Mapping):
            val = tf214.get("fit_valid")
        else:
            val = getattr(getattr(tf214, "params", {}), "get", lambda *a: None)("fit_valid")
        diagnostics["time_fit_po214_fit_valid"] = bool(val) if val is not None else None

    tf218 = time_fit_results.get("Po218") if time_fit_results else None
    if tf218 is not None:
        if isinstance(tf218, Mapping):
            val = tf218.get("fit_valid")
        else:
            val = getattr(getattr(tf218, "params", {}), "get", lambda *a: None)("fit_valid")
        diagnostics["time_fit_po218_fit_valid"] = bool(val) if val is not None else None

    removed_noise = summary.get("noise_cut", {}).get("removed_events", 0)
    removed_burst = summary.get("burst_filter", {}).get("removed_events", 0)
    n_loaded = len(df_analysis) + int(removed_noise) + int(removed_burst)
    diagnostics["n_events_loaded"] = int(n_loaded)
    diagnostics["n_events_discarded"] = int(n_loaded - len(df_analysis))

    bin_mode = str(
        cfg.get("plot_time_binning_mode", cfg.get("time_bin_mode", "fixed"))
    ).lower()
    bin_width = None
    if bin_mode not in ("fd", "auto"):
        bin_width = cfg.get("plot_time_bin_width_s", cfg.get("time_bin_s"))

    diagnostics["selected_analysis_modes"] = {
        "background_model": summary.get("analysis", {}).get("background_model"),
        "spectral_fit": {
            "unbinned_likelihood": bool(
                cfg.get("spectral_fit", {}).get("unbinned_likelihood", False)
            )
        },
        "plotting": {
            "time_bin_mode": bin_mode,
            "time_bin_width_s": bin_width,
        },
    }

    diagnostics["warnings"] = get_captured_warnings()

    # Baseline diagnostics (optional)
    if events_df is not None and calibration is not None:
        baseline_info = summary.get("baseline", {})
        if baseline_info and cfg.get("generate_baseline_diagnostics", False):
            try:
                from baseline_diagnostics import build_baseline_diagnostics

                baseline_diag = build_baseline_diagnostics(
                    events=events_df,
                    baseline_info=baseline_info,
                    calibration=calibration,
                    output_dir=output_dir,
                )
                diagnostics["baseline_quality"] = baseline_diag
            except Exception as e:
                diagnostics["warnings"].append(
                    f"Baseline diagnostics generation failed: {str(e)}"
                )

    return diagnostics


__all__ = [
    "DEFAULT_DIAGNOSTICS",
    "start_warning_capture",
    "get_captured_warnings",
    "build_diagnostics",
]
