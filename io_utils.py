# io_utils.py
from pathlib import Path
import shutil
import json
import yaml
import logging
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from dateutil import parser as date_parser
import argparse
import pandas as pd
from collections.abc import Mapping
from typing import Any, Iterator
from constants import load_nuclide_overrides

import numpy as np
from utils import to_native
from utils.time_utils import parse_timestamp, to_epoch_seconds, tz_convert_utc
import jsonschema
from reporting import DEFAULT_DIAGNOSTICS
from config.validation import validate_radon_inference


def extract_time_series_events(events, cfg):
    """Slice events for time-series fits based on isotope windows.

    Configuration keys **must** use lowercase isotope names, for example
    (``window_po214`` etc.). Mixed-case keys such as
    ``window_Po214`` are still recognized for backward
    compatibility.

    Parameters
    ----------
    events : pandas.DataFrame
        Event data with ``timestamp`` and ``energy_MeV`` columns.
    cfg : dict
        Configuration containing ``time_fit`` settings. The window
        definitions should use lowercase keys.

    Returns
    -------
    dict
        Mapping of isotope name to ``numpy.ndarray`` of timestamps.
    """

    ts_cfg = cfg.get("time_fit", {})
    out = {}
    for iso in ("Po214", "Po218", "Po210"):
        # Windows are stored using lowercase isotope names
        win = ts_cfg.get(f"window_{iso.lower()}")
        if win is None:
            continue
        lo, hi = win
        mask = (events["energy_MeV"] >= lo) & (events["energy_MeV"] <= hi)
        out[iso] = events.loc[mask, "timestamp"].to_numpy()
    return out


logger = logging.getLogger(__name__)


@dataclass
class Summary(Mapping[str, Any]):
    """Summary information written to ``summary.json``."""

    timestamp: str | None = None
    config_used: str | None = None
    config_sha256: str | None = None
    calibration: dict = field(default_factory=dict)
    calibration_valid: bool | None = None
    spectral_fit: dict = field(default_factory=dict)
    time_fit: dict = field(default_factory=dict)
    systematics: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    radon_results: dict = field(default_factory=dict)
    radon_combined: dict = field(default_factory=dict)
    radon: dict = field(default_factory=dict)
    po214: dict = field(default_factory=dict)
    po218: dict = field(default_factory=dict)
    noise_cut: dict = field(default_factory=dict)
    burst_filter: dict = field(default_factory=dict)
    adc_drift_rate: float | None = None
    adc_drift_mode: str | None = None
    adc_drift_params: dict = field(default_factory=dict)
    efficiency: dict = field(default_factory=dict)
    random_seed: int | None = None
    git_commit: str | None = None
    requirements_sha256: str | None = None
    cli_sha256: str | None = None
    cli_args: list[str] = field(default_factory=list)
    analysis: dict = field(default_factory=dict)
    diagnostics: dict | None = None

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(asdict(self))

    def __len__(self) -> int:  # type: ignore[override]
        return len(asdict(self))

    def get(self, key: str, default=None) -> Any:
        return getattr(self, key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pipeline": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "log_level": {"type": "string"},
                "random_seed": {"type": ["integer", "null"]},
            },
            "required": ["log_level"],
        },
        "allow_fallback": {"type": "boolean"},
        "allow_negative_baseline": {"type": "boolean"},
        "allow_negative_activity": {"type": "boolean"},
        "analysis_isotope": {"type": "string", "enum": ["radon", "po218", "po214"]},
        "spectral_fit": {
            "type": "object",
            "properties": {"expected_peaks": {"type": "object"}},
        },
        "time_fit": {
            "type": "object",
            "properties": {
                "do_time_fit": {"type": "boolean"},
                "hl_po214": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 2,
                },
                "hl_po218": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 2,
                },
                "hl_po210": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 2,
                },
            },
            "required": ["do_time_fit"],
        },
        "systematics": {
            "type": "object",
            "properties": {"enable": {"type": "boolean"}},
            "required": ["enable"],
        },
        "plotting": {
            "type": "object",
            "properties": {"plot_save_formats": {"type": "array"}},
            "required": ["plot_save_formats"],
        },
        "fitting": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "use_stable_emg": {"type": "boolean"},
                "emg_stable_mode": {
                    "type": ["string", "boolean"],
                    "enum": [
                        True, False,
                        "", "auto", "default",
                        "scipy_safe", "erfcx", "erfcx_exact",
                        "legacy", "direct", "stable",
                        "exponnorm", "off", "disabled", "scipy",
                    ],
                },
                "emg_tau_min": {"type": "number", "exclusiveMinimum": 0},
                "per_isotope": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "use_emg": {"type": ["boolean", "object"]},
                            "tau": {"type": "number"},
                            "tau_prior": {"type": "array", "minItems": 2, "maxItems": 2},
                            "peak_sigma": {"type": "number"},
                            "peak_scale": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
        },
        "baseline": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "range": {
                    "type": "array",
                    "items": {"type": ["string", "number"]},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "monitor_volume_l": {"type": "number", "minimum": 0},
                "sample_volume_l": {"type": "number", "minimum": 0},
                "isotopes_to_subtract": {"type": "array", "items": {"type": "string"}},
            },
        },
        "burst_filter": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "burst_mode": {"type": "string"},
                "burst_window_size_s": {"type": "number", "minimum": 0},
                "rolling_median_window": {"type": "number", "minimum": 0},
                "burst_multiplier": {"type": "number", "minimum": 0},
                "micro_window_size_s": {"type": "number", "minimum": 0},
                "micro_count_threshold": {"type": "number", "minimum": 0},
            },
        },
        "calibration": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "noise_cutoff": {"type": "number", "minimum": 0},
                "hist_bins": {"type": "integer", "minimum": 1},
                "peak_search_radius": {"type": "number", "minimum": 0},
                "peak_prominence": {"type": "number", "minimum": 0},
                "peak_width": {"type": "number", "minimum": 0},
                "nominal_adc": {"type": "object"},
                "fit_window_adc": {"type": "number", "minimum": 0},
                "use_emg": {"type": "boolean"},
                "float_slope": {"type": "boolean"},
                "init_sigma_adc": {"type": "number", "minimum": 0},
                "init_tau_adc": {"type": "number", "minimum": 0},
                "sanity_tolerance_mev": {"type": "number", "minimum": 0},
                "known_energies": {"type": "object"},
                "use_quadratic": {"type": ["string", "boolean"]},
            },
        },
        "analysis": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "analysis_start_time": {"type": ["string", "number", "null"]},
                "analysis_end_time": {"type": ["string", "number", "null"]},
                "spike_start_time": {"type": ["string", "number", "null"]},
                "spike_end_time": {"type": ["string", "number", "null"]},
                "spike_periods": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "array",
                        "items": {"type": ["string", "number"]},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "run_periods": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": ["string", "number"]},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "radon_interval": {
                    "type": "array",
                    "items": {"type": ["string", "number"]},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "ambient_concentration": {"type": ["number", "null"]},
                "settle_s": {"type": ["number", "null"], "minimum": 0},
                "background_model": {
                    "type": "string",
                    "enum": ["linear", "loglin_unit"],
                },
                "likelihood": {
                    "type": "string",
                    "enum": ["current", "extended"],
                },
            },
        },
        "columns": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "fUniqueID": {"type": "string"},
                "fBits": {"type": "string"},
                "timestamp": {"type": "string"},
                "adc": {"type": "string"},
                "fchannel": {"type": "string"},
            },
        },
        "radon_inference": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean"},
                "source_isotopes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["Po214", "Po218"],
                    },
                    "minItems": 1,
                },
                "source_weights": {
                    "type": "object",
                    "patternProperties": {
                        "^(Po214|Po218)$": {"type": "number"}
                    },
                    "additionalProperties": False,
                },
                "detection_efficiency": {
                    "type": "object",
                    "patternProperties": {
                        "^(Po214|Po218)$": {"type": "number"}
                    },
                    "additionalProperties": False,
                },
                "transport_efficiency": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1.5,
                },
                "retention_efficiency": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1.5,
                },
                "chain_correction": {
                    "type": "string",
                    "enum": ["none", "assume_equilibrium", "forward_model"],
                },
                "external_rn": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["constant", "file"],
                        },
                        "constant_bq_per_m3": {"type": "number"},
                        "fallback_bq_per_m3": {"type": "number"},
                        "file_path": {"type": "string"},
                        "time_column": {"type": "string"},
                        "value_column": {"type": "string"},
                        "tz": {"type": "string"},
                        "interpolation": {
                            "type": "string",
                            "enum": ["nearest", "ffill"],
                        },
                        "allowed_skew_seconds": {"type": "number"},
                        "max_gap_seconds": {"type": "number"},
                    },
                    "required": ["mode"],
                    "allOf": [
                        {
                            "if": {"properties": {"mode": {"const": "constant"}}},
                            "then": {"required": ["constant_bq_per_m3"]},
                        },
                        {
                            "if": {"properties": {"mode": {"const": "file"}}},
                            "then": {"required": ["file_path"]},
                        },
                    ],
                },
                "output": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "write_per_interval": {"type": "boolean"},
                        "write_cumulative": {"type": "boolean"},
                    },
                },
            },
            "required": ["enabled"],
        },
        "efficiency": {"type": "object"},
    },
    "required": [
        "pipeline",
        "spectral_fit",
        "time_fit",
        "systematics",
        "plotting",
    ],
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key '{key}' in configuration")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def ensure_dir(path):
    """Create directory if it does not exist."""
    p = Path(path)
    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)


# parse_timestamp is provided by :mod:`utils.time_utils` and re-exported from
# :mod:`utils` for backward compatibility with older code.


def load_config(config_path):
    """Load a configuration mapping or YAML file and validate it."""

    if isinstance(config_path, Mapping):
        cfg = dict(config_path)
    else:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError("Config file must be YAML")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=_UniqueKeyLoader) or {}

    validator = jsonschema.Draft7Validator(CONFIG_SCHEMA)
    missing = []
    for err in validator.iter_errors(cfg):
        if err.validator == "required":
            key = err.message.split("'")[1]
            dotted = ".".join(list(err.absolute_path) + [key])
            missing.append(dotted)
    if missing:
        raise ValueError("Missing required keys: " + ", ".join(missing))

    try:
        jsonschema.validate(cfg, CONFIG_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        if e.validator == "required":
            raise KeyError(e.message)
        raise

    validate_radon_inference(cfg)

    cfg["nuclide_constants"] = load_nuclide_overrides(cfg)

    if "analysis_isotope" not in cfg:
        cfg["analysis_isotope"] = "radon"

    fit_cfg_raw = cfg.get("fitting")
    if fit_cfg_raw is None:
        fit_cfg: dict[str, Any] = {}
        cfg["fitting"] = fit_cfg
    elif isinstance(fit_cfg_raw, Mapping) and not isinstance(fit_cfg_raw, dict):
        fit_cfg = dict(fit_cfg_raw)
        cfg["fitting"] = fit_cfg
    elif isinstance(fit_cfg_raw, dict):
        fit_cfg = fit_cfg_raw
    else:
        raise TypeError("'fitting' section must be a mapping if provided")

    use_stable_emg_raw = fit_cfg.get("use_stable_emg")
    if use_stable_emg_raw is None:
        use_stable_emg = True
    else:
        use_stable_emg = bool(use_stable_emg_raw)

    emg_stable_mode_raw = fit_cfg.get("emg_stable_mode")
    if emg_stable_mode_raw is None:
        emg_stable_mode = use_stable_emg
    else:
        emg_stable_mode = bool(emg_stable_mode_raw)
        use_stable_emg = emg_stable_mode

    fit_cfg["use_stable_emg"] = use_stable_emg
    fit_cfg["emg_stable_mode"] = emg_stable_mode

    tau_min_raw = fit_cfg.get("emg_tau_min")
    tau_min = float(5.0e-4 if tau_min_raw is None else tau_min_raw)
    fit_cfg["emg_tau_min"] = tau_min

    import constants as _constants

    _constants._TAU_MIN = tau_min

    # Update centralized EMG constants
    try:
        import rmtest.emg_constants as _emg_constants
        # Note: Module-level constants are read-only, but we can update via setattr
        # This allows runtime configuration to override defaults
        object.__setattr__(_emg_constants, 'EMG_MIN_TAU', tau_min)
    except (ImportError, AttributeError):  # pragma: no cover
        pass

    try:
        import calibration as _calibration  # type: ignore

        if hasattr(_calibration, "configure_emg"):
            _calibration.configure_emg(use_stable_emg, tau_min)
        else:  # pragma: no cover - compatibility with older versions
            _calibration.USE_STABLE_EMG = use_stable_emg
            if hasattr(_calibration, "_set_tau_min"):
                _calibration._set_tau_min(tau_min)
            else:
                _calibration._TAU_MIN = tau_min
    except ImportError:  # pragma: no cover - calibration module optional in some contexts
        pass

    try:
        import fitting as _fitting  # type: ignore

        _fitting._TAU_MIN = tau_min
        if hasattr(_fitting, "_update_emg_stable_mode_from_config"):
            _fitting._update_emg_stable_mode_from_config(cfg)
        else:  # pragma: no cover - fallback for older versions without helper
            _fitting.EMG_STABLE_MODE = emg_stable_mode
    except ImportError:  # pragma: no cover - fitting may be optional in some contexts
        pass

    # Fill in default EMG usage for spectral fits honoring tau priors and overrides
    spec = cfg.setdefault("spectral_fit", {})
    isotopes = ("Po210", "Po218", "Po214")
    use_emg = {iso: False for iso in isotopes}
    forced_true: set[str] = set()
    explicit: set[str] = set()

    for iso in isotopes:
        mean_key = f"tau_{iso}_prior_mean"
        sigma_key = f"tau_{iso}_prior_sigma"
        if (
            mean_key in spec
            and sigma_key in spec
            and spec[mean_key] is not None
            and spec[sigma_key] is not None
        ):
            use_emg[iso] = True
            forced_true.add(iso)
            explicit.add(iso)

    emg_cfg = spec.get("use_emg")
    if isinstance(emg_cfg, Mapping):
        for iso, value in emg_cfg.items():
            if iso in use_emg and iso not in forced_true:
                use_emg[iso] = bool(value)
                explicit.add(iso)
    elif emg_cfg is not None:
        scalar: bool | None
        if isinstance(emg_cfg, bool):
            scalar = emg_cfg
        else:
            scalar = bool(emg_cfg)
        for iso in isotopes:
            if iso in forced_true or iso in explicit:
                continue
            use_emg[iso] = scalar

    spec["use_emg"] = use_emg
    # Validate resolution settings
    float_sigma_E = spec.get("float_sigma_E", True)
    fix_sigma0 = spec.get("flags", {}).get("fix_sigma0", False)
    if float_sigma_E and fix_sigma0:
        raise ValueError(
            "Resolution flags conflict: cannot float energy resolution while fixing sigma0"
        )
    # CONFIG_SCHEMA validation already checks required keys

    return cfg


def load_events(csv_path, *, start=None, end=None, column_map=None):
    """
    Read event CSV into a DataFrame with columns:
       ['fUniqueID','fBits','timestamp','adc','fchannel']
    Column aliases like ``time`` or ``adc_ch`` are automatically renamed to
    their canonical form.  A mapping of canonical column names to the
    actual CSV headers may be supplied via ``column_map``. The ``timestamp``
    column is parsed to ``datetime64[ns, UTC]`` while ``adc`` is returned as a
    floating point number. The DataFrame is sorted by ``timestamp`` before being
    returned.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Read CSV strictly as strings to avoid type inference surprises
    df = pd.read_csv(path, sep=",", engine="c", dtype=str)

    # Rename columns based on explicit configuration mapping
    if column_map:
        cfg_rename = {v: k for k, v in column_map.items() if v in df.columns}
        if cfg_rename:
            df = df.rename(columns=cfg_rename, errors="ignore")

    # Allow some common alternate column names
    rename = {
        "time": "timestamp",
        "adc_ch": "adc",
        "adc_channel": "adc",
        "channel": "fchannel",
        "unique_id": "fUniqueID",
        "bits": "fBits",
    }

    df = df.rename(columns=rename, errors="ignore")

    # Parse timestamps directly to timezone-aware ``Timestamp`` values
    if "timestamp" in df.columns:

        def _safe_parse(val):
            try:
                return parse_timestamp(val)
            except Exception:
                return pd.NaT

        df["timestamp"] = df["timestamp"].map(_safe_parse)

    # Check required columns after renaming
    required_cols = ["fUniqueID", "fBits", "timestamp", "adc", "fchannel"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}")

    # Convert numeric columns explicitly
    df["adc"] = pd.to_numeric(df.get("adc"), errors="coerce")

    start_len = len(df)

    # Drop rows with invalid values that became NaN
    df = df.dropna(subset=["timestamp", "adc"])
    mask = np.isfinite(df["adc"])
    df = df[mask]

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    discarded = start_len - len(df)

    # Convert types
    df["adc"] = df["adc"].astype(float)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    if start is not None:
        start_dt = parse_timestamp(start)
        df = df[df["timestamp"] >= start_dt]
    if end is not None:
        end_dt = parse_timestamp(end)
        df = df[df["timestamp"] <= end_dt]

    frac_discarded = discarded / start_len if start_len > 0 else 0
    logger.info(
        f"Loaded {len(df)} events from {csv_path} "
        f"({discarded} discarded, {frac_discarded:.1%})."
    )
    return df


def apply_burst_filter(df, cfg=None, mode="rate"):
    """Remove events occurring during high-rate bursts.

    ``mode`` selects the filtering strategy:

    ``'none'`` – return the input unchanged;
    ``'micro'`` – short sliding-window burst veto;
    ``'rate'`` – rolling-median rate-based veto (legacy behaviour);
    ``'both'`` – apply the micro window first then the rate veto.

    Parameters
    ----------
    df : pandas.DataFrame
        Event data containing a ``timestamp`` column. Values may be numeric
        epoch seconds or ``datetime64`` objects. All timestamps are first
        converted to ``datetime64`` using :func:`parse_timestamp` and the
        DataFrame is updated accordingly. Seconds are only used internally for
        histogram calculations.
    cfg : dict, optional
        Configuration dictionary. Expected keys under ``burst_filter`` are
        ``burst_window_size_s``, ``rolling_median_window`` and
        ``burst_multiplier``.

    Returns
    -------
    DataFrame
        Filtered events with bursts removed.
    int
        Number of events removed.
    """

    cfg = cfg or {}
    bcfg = cfg.get("burst_filter", {})
    warnings.filterwarnings(
        "ignore",
        message=".*np.find_common_type.*",
        category=DeprecationWarning,
    )

    if mode == "none" or len(df) == 0:
        return df.copy(), 0

    removed_total = 0
    out_df = df.copy()

    ts = out_df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = ts.map(parse_timestamp)
    else:
        if ts.dt.tz is None:
            ts = ts.map(parse_timestamp)
        else:
            ts = tz_convert_utc(ts)
    out_df["timestamp"] = ts
    times_sec = ts.map(to_epoch_seconds).to_numpy()

    # ───── micro-burst veto ─────
    if mode in ("micro", "both"):
        micro_win = bcfg.get("micro_window_size_s")
        micro_thr = bcfg.get("micro_count_threshold")

        if micro_win is not None and micro_thr is not None:
            times = times_sec
            if len(times) == 0:
                return out_df, removed_total

            t_min = times.min()
            t_max = times.max()
            hist, edges = np.histogram(times, bins=np.arange(t_min, t_max + 2))
            hist = hist.astype(int)

            win = int(micro_win)
            thr = int(micro_thr)
            if win > 0:
                csum = np.concatenate([[0], np.cumsum(hist)])
                counts = csum[win:] - csum[:-win]
                burst_bins = np.zeros_like(hist, dtype=bool)
                for i, c in enumerate(counts):
                    if c >= thr:
                        burst_bins[i : i + win] = True
            else:
                burst_bins = np.zeros_like(hist, dtype=bool)

            bin_idx = np.searchsorted(edges, times, side="right") - 1
            to_remove = burst_bins[bin_idx]

            removed_total += int(to_remove.sum())
            out_df = out_df[~to_remove].reset_index(drop=True)

            # Recalculate times after removing events
            ts = out_df["timestamp"]
            if not pd.api.types.is_datetime64_any_dtype(ts):
                ts = ts.map(parse_timestamp)
            else:
                if ts.dt.tz is None:
                    ts = ts.map(parse_timestamp)
                else:
                    ts = tz_convert_utc(ts)
            out_df["timestamp"] = ts
            times_sec = ts.map(to_epoch_seconds).to_numpy()

    # ───── rate-based veto ─────
    if mode in ("rate", "both"):
        win = bcfg.get("burst_window_size_s")
        roll = bcfg.get("rolling_median_window")
        mult = bcfg.get("burst_multiplier")

        if (
            win is not None
            and roll is not None
            and mult is not None
            and len(out_df) > 0
        ):
            t0 = times_sec.min()
            bins = pd.Series(
                ((times_sec - t0) // float(win)).astype(int), index=out_df.index
            )

            counts = bins.value_counts().sort_index()
            full_index = range(counts.index.min(), counts.index.max() + 1)
            counts_full = counts.reindex(full_index, fill_value=0)

            med = counts_full.rolling(int(roll), center=True, min_periods=1).median()

            threshold = mult * med
            burst_bins = counts_full.astype(float)[
                counts_full.astype(float) > threshold
            ].index.to_numpy(dtype=np.int64)
            mask = ~bins.isin(burst_bins)

            removed_total += int((~mask).sum())
            out_df = out_df[mask].reset_index(drop=True)

    return out_df, removed_total


def write_summary(
    output_dir: str | Path,
    summary_dict: Mapping[str, Any] | Summary,
    timestamp: str | None = None,
) -> Path:
    """Write ``summary_dict`` to ``summary.json`` and return the results folder."""

    output_path = Path(output_dir)

    if timestamp is None and output_path.is_dir():
        results_folder = output_path
    else:
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        results_folder = output_path / timestamp
        if results_folder.exists():
            raise FileExistsError(f"Results folder already exists: {results_folder}")
        results_folder.mkdir(parents=True, exist_ok=False)

    summary_path = results_folder / "summary.json"

    sanitized = to_native(summary_dict)

    if "diagnostics" not in sanitized or sanitized["diagnostics"] is None:
        sanitized["diagnostics"] = to_native(DEFAULT_DIAGNOSTICS)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=4)

    logger.info(f"Wrote summary JSON to {summary_path}")
    return results_folder


def copy_config(output_dir, config_path, *, exist_ok=False):
    """
    Copy the used config JSON into the timestamped results folder.

    Parameters
    ----------
    output_dir : Path or str
        Path to the directory where ``config_used.json`` should be placed.
        The directory will be created if needed.
    exist_ok : bool, optional
        If ``True``, allow ``output_dir`` to already exist.
        Defaults to ``False``.
    config_path : Path, str or dict
        Configuration file to copy or configuration dictionary.

    Returns
    -------
    Path
        Destination of the copied config.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=exist_ok)

    dest_path = output_path / "config_used.json"
    if isinstance(config_path, (str, Path)):
        shutil.copyfile(Path(config_path), dest_path)
        logger.info(f"Copied config {config_path} -> {dest_path}")
    else:
        sanitized = to_native(config_path)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=4)
        logger.info(f"Wrote config to {dest_path}")
    return dest_path
