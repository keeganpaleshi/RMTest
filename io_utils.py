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
from config.validation import validate_lucas_bridge, validate_radon_inference


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
    radon_inference: dict = field(default_factory=dict)
    radon: dict = field(default_factory=dict)
    po214: dict = field(default_factory=dict)
    po218: dict = field(default_factory=dict)
    noise_cut: dict = field(default_factory=dict)
    burst_filter: dict = field(default_factory=dict)
    downtime: dict = field(default_factory=dict)
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
    lucas_bridge: dict = field(default_factory=dict)
    template_fitting: dict = field(default_factory=dict)

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


_DEFAULT_LOGLIN_NORM_SAMPLES = 512
_DEFAULT_MAX_B1_SIGMA = 10.0


@dataclass(frozen=True)
class _KeyRule:
    internal: str
    canonical: str
    aliases: tuple[str, ...] = ()


_CONFIG_KEY_RULES: dict[tuple[str, ...], tuple[_KeyRule, ...]] = {
    ("columns",): (
        _KeyRule("fUniqueID", "unique_id", ("funique_id",)),
        _KeyRule("fBits", "bits", ("fbits",)),
        _KeyRule("fchannel", "channel"),
    ),
    ("calibration",): (
        _KeyRule("slope_MeV_per_ch", "slope_mev_per_ch"),
        _KeyRule("sigma_E_init", "sigma_e_init"),
        _KeyRule("intercept_MeV", "intercept_mev"),
    ),
    ("spectral_fit",): (
        _KeyRule("loglin_n_norm", "background_norm_points"),
        _KeyRule("S_bkg_prior", "s_bkg_prior"),
        _KeyRule("sigma_E_prior_source", "sigma_e_prior_source"),
        _KeyRule("sigma_E_prior_sigma", "sigma_e_prior_sigma"),
        _KeyRule("sigma_E_prior_mean", "sigma_e_prior_mean"),
        _KeyRule("float_sigma_E", "float_sigma_e"),
        _KeyRule("tau_Po210_prior_mean", "tau_po210_prior_mean"),
        _KeyRule("tau_Po210_prior_sigma", "tau_po210_prior_sigma"),
        _KeyRule("tau_Po218_prior_mean", "tau_po218_prior_mean"),
        _KeyRule("tau_Po218_prior_sigma", "tau_po218_prior_sigma"),
        _KeyRule("tau_Po214_prior_mean", "tau_po214_prior_mean"),
        _KeyRule("tau_Po214_prior_sigma", "tau_po214_prior_sigma"),
        _KeyRule("tau_Po216_prior_mean", "tau_po216_prior_mean"),
        _KeyRule("tau_Po216_prior_sigma", "tau_po216_prior_sigma"),
        _KeyRule("tau_Po212_prior_mean", "tau_po212_prior_mean"),
        _KeyRule("tau_Po212_prior_sigma", "tau_po212_prior_sigma"),
    ),
    ("spectral_fit", "flags"): (
        _KeyRule("fix_F", "fix_f"),
        _KeyRule("F_prior", "f_prior"),
    ),
    ("time_fit",): (
        _KeyRule("window_po210", "window_po210", ("window_Po210",)),
        _KeyRule("window_po218", "window_po218", ("window_Po218",)),
        _KeyRule("window_po214", "window_po214", ("window_Po214",)),
    ),
    ("time_fit", "flags"): (
        _KeyRule("fix_N0_po210", "fix_n0_po210", ("fix_N0_Po210",)),
        _KeyRule("fix_N0_po218", "fix_n0_po218", ("fix_N0_Po218",)),
        _KeyRule("fix_N0_po214", "fix_n0_po214", ("fix_N0_Po214",)),
    ),
    ("systematics",): (
        _KeyRule("sigma_E_frac", "sigma_e_frac"),
        _KeyRule("energy_shift_keV", "energy_shift_kev"),
    ),
}


def _rule_candidates(rule: _KeyRule) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in (rule.canonical, *rule.aliases, rule.internal):
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return tuple(ordered)


def _format_config_path(path: tuple[str, ...], key: str) -> str:
    return ".".join((*path, key)) if path else key


def _coalesce_section_keys(section: dict[str, Any], path: tuple[str, ...]) -> None:
    for rule in _CONFIG_KEY_RULES.get(path, ()):
        chosen_key = None
        chosen_value = None
        for candidate in _rule_candidates(rule):
            if candidate not in section:
                continue
            if chosen_key is None:
                chosen_key = candidate
                chosen_value = section[candidate]
                continue
            if section[candidate] != chosen_value:
                logger.warning(
                    "Config sets both %s and %s; preferring %s.",
                    _format_config_path(path, chosen_key),
                    _format_config_path(path, candidate),
                    _format_config_path(path, chosen_key),
                )
        if chosen_key is None:
            continue
        section[rule.internal] = chosen_value
        for candidate in _rule_candidates(rule):
            if candidate != rule.internal:
                section.pop(candidate, None)


def _normalize_config_key_aliases(obj: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(obj, Mapping):
        normalized = dict(obj)
        _coalesce_section_keys(normalized, path)
        for key, value in list(normalized.items()):
            next_path = path + (key,) if isinstance(key, str) else path
            normalized[key] = _normalize_config_key_aliases(value, next_path)
        return normalized
    if isinstance(obj, list):
        return [_normalize_config_key_aliases(item, path) for item in obj]
    return obj


def _canonicalize_config_keys(obj: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(obj, Mapping):
        rules = {rule.internal: rule.canonical for rule in _CONFIG_KEY_RULES.get(path, ())}
        canonicalized: dict[str, Any] = {}
        for key, value in obj.items():
            canonical_key = rules.get(key, key)
            next_path = path + (canonical_key,) if isinstance(canonical_key, str) else path
            canonicalized[canonical_key] = _canonicalize_config_keys(value, next_path)
        return canonicalized
    if isinstance(obj, list):
        return [_canonicalize_config_keys(item, path) for item in obj]
    return obj


_SUMMARY_KEY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Rn_activity", "rn_activity"),
    ("sigma_E", "sigma_e"),
    ("S_bkg", "s_bkg"),
    ("_UL95_", "_ul95_"),
    ("_Bq_per_L", "_bq_per_l"),
    ("_Bq_per_m3", "_bq_per_m3"),
    ("_Bq", "_bq"),
    ("_MeV", "_mev"),
    ("_keV", "_kev"),
)


def _canonicalize_summary_key(key: str) -> str:
    canonical = key
    for old, new in _SUMMARY_KEY_REPLACEMENTS:
        canonical = canonical.replace(old, new)
    return canonical


def _with_summary_key_aliases(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        aliased: dict[Any, Any] = {}
        for key, value in obj.items():
            converted = _with_summary_key_aliases(value)
            if not isinstance(key, str):
                aliased[key] = converted
                continue
            canonical_key = _canonicalize_summary_key(key)
            aliased.setdefault(canonical_key, converted)
            if canonical_key != key:
                aliased.setdefault(key, converted)
        return aliased
    if isinstance(obj, list):
        return [_with_summary_key_aliases(item) for item in obj]
    return obj


CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
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
            "properties": {
                "expected_peaks": {"type": "object"},
                "clip_floor": {
                    "type": "number",
                    "exclusiveMinimum": 0.0,
                    "maximum": 1e-6,
                    "description": "Small positive floor applied to per-E PDFs to avoid log(0); must be tiny."
                },
            },
        },
        "time_fit": {
            "type": "object",
            # IMPORTANT: keep this True so older configs and test merge paths don't die
            "additionalProperties": True,
            "properties": {
                "model": {
                    "type": "string",
                    "enum": [
                        "single_exp",
                        "bi_exp",
                        "constant",
                        "poisson_decays",
                    ],
                },
                "t0": {"type": "number"},
                "fix_lambda": {"type": "boolean"},
                "lambda": {"type": "number"},
                "activity_units": {
                    "type": "string",
                    "enum": ["Bq", "cpd"],
                },
                "do_time_fit": {"type": "boolean"},
                "hl_po214": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0},
                        {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0.0},
                        },
                        {"type": "null"},
                    ]
                },
                "hl_po218": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0},
                        {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0.0},
                        },
                        {"type": "null"},
                    ]
                },
                "hl_po210": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0},
                        {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0.0},
                        },
                        {"type": "null"},
                    ]
                },
            },
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
                "mode": {
                    "type": "string",
                    "enum": ["auto", "linear", "manual"],
                },
                "allow_negative_baseline": {"type": "boolean"},
                "b0_prior": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "b1_prior": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
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
                    "type": ["array", "null"],
                    "items": {"type": ["string", "number"]},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "plot_radon_activity": {"type": "boolean"},
                "ambient_concentration": {"type": ["number", "null"]},
                "settle_s": {"type": ["number", "null"], "minimum": 0},
                "background_model": {
                    "type": "string",
                    "enum": ["linear", "loglin_unit", "sigmoid_unit", "exp_unit", "double_logit_unit", "none"],
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
                        "time_columns": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "year": {"type": "string"},
                                "month": {"type": "string"},
                                "day": {"type": "string"},
                                "hour": {"type": "string"},
                                "minute": {"type": "string"},
                                "year_format": {
                                    "type": "string",
                                    "enum": ["two_digit"],
                                },
                            },
                        },
                        "units": {
                            "type": "string",
                            "enum": ["bq_per_m3", "pci_per_l"],
                        },
                        "tz": {"type": "string"},
                        "timezone": {"type": "string"},
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
        "lucas_bridge": {"type": "object"},
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
        if path.suffix in {".yaml", ".yml"}:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=_UniqueKeyLoader) or {}
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
        else:
            raise ValueError("Config file must be YAML or JSON")

    cfg = _normalize_config_key_aliases(cfg)

    sf = cfg.setdefault("spectral_fit", {})
    analysis_cfg = cfg.setdefault("analysis", {})
    legacy_background_model = sf.get("bkg_mode")
    if isinstance(legacy_background_model, str):
        legacy_background_model = legacy_background_model.strip().lower()
        if legacy_background_model in {"linear", "loglin_unit"}:
            explicit_background_model = analysis_cfg.get("background_model")
            if explicit_background_model is None:
                analysis_cfg["background_model"] = legacy_background_model
                logger.warning(
                    "Config uses legacy spectral_fit.bkg_mode=%r as a background model selector; "
                    "normalizing to analysis.background_model=%r and spectral_fit.bkg_mode='manual'.",
                    legacy_background_model,
                    legacy_background_model,
                )
            elif explicit_background_model != legacy_background_model:
                logger.warning(
                    "Config sets conflicting background models: analysis.background_model=%r "
                    "takes precedence over legacy spectral_fit.bkg_mode=%r.",
                    explicit_background_model,
                    legacy_background_model,
                )
            sf["bkg_mode"] = "manual"

    # Pre-validate clip_floor so tests see a ValueError (not a schema error).
    # Do this BEFORE jsonschema.validate, otherwise the schema's
    # exclusiveMinimum/maximum will throw first.
    if "clip_floor" in sf:
        try:
            sf["clip_floor"] = float(sf["clip_floor"])
        except Exception:
            raise ValueError("spectral_fit.clip_floor must be in (0, 1e-6].")
        if not (0.0 < sf["clip_floor"] <= 1e-6):
            raise ValueError("spectral_fit.clip_floor must be in (0, 1e-6].")
    else:
        sf["clip_floor"] = 1e-300

    if "loglin_n_norm" in sf:
        try:
            sf["loglin_n_norm"] = int(sf["loglin_n_norm"])
        except Exception as exc:
            raise ValueError("spectral_fit.background_norm_points (legacy spectral_fit.loglin_n_norm) must be a positive integer") from exc
        if sf["loglin_n_norm"] <= 0:
            raise ValueError("spectral_fit.background_norm_points (legacy spectral_fit.loglin_n_norm) must be a positive integer")
    else:
        sf["loglin_n_norm"] = _DEFAULT_LOGLIN_NORM_SAMPLES

    max_b1_sigma = sf.get("max_b1_sigma", _DEFAULT_MAX_B1_SIGMA)
    try:
        max_b1_sigma = float(max_b1_sigma)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("spectral_fit.max_b1_sigma must be positive") from exc
    if max_b1_sigma <= 0:
        raise ValueError("spectral_fit.max_b1_sigma must be positive")
    sf["max_b1_sigma"] = max_b1_sigma

    if "b1_prior" in sf:
        prior = sf["b1_prior"]
        if not isinstance(prior, (list, tuple)) or len(prior) < 2:
            raise ValueError("spectral_fit.b1_prior must be [mean, sigma]")
        mean, sigma = prior[0], prior[1]
        sigma = float(sigma)
        mean = float(mean)
        if sigma <= 0:
            raise ValueError("spectral_fit.b1_prior sigma must be positive")
        if sigma > max_b1_sigma:
            warnings.warn(
                "spectral_fit.b1_prior sigma exceeds max_b1_sigma; clamping to limit",
                RuntimeWarning,
            )
            sigma = max_b1_sigma
        sf["b1_prior"] = (mean, sigma)

    validator = jsonschema.Draft7Validator(CONFIG_SCHEMA)
    missing = []
    for err in validator.iter_errors(cfg):
        if err.validator == "required":
            # Safely extract key from error message
            parts = err.message.split("'")
            if len(parts) >= 2:
                key = parts[1]
            else:
                # Fallback: extract from the message or use a placeholder
                key = err.message.split()[-1] if err.message else "unknown"
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
    validate_lucas_bridge(cfg)

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
    isotopes = ("Po210", "Po218", "Po214", "Po216", "Po212")
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
        canonical_column_keys = {
            "unique_id": "fUniqueID",
            "bits": "fBits",
            "channel": "fchannel",
        }
        normalized_column_map = {
            canonical_column_keys.get(key, key): value for key, value in column_map.items()
        }
        cfg_rename = {v: k for k, v in normalized_column_map.items() if v in df.columns}
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
        ts_raw = df["timestamp"]

        # Fast path for numeric epoch seconds (either native numbers or strings)
        ts_numeric = pd.to_numeric(ts_raw, errors="coerce")
        parsed = pd.to_datetime(ts_numeric, unit="s", utc=True, errors="coerce")

        # ``pd.to_datetime`` with ``utc=True`` already handles naive or
        # timezone-aware ISO strings. Only fall back to that slower parser for
        # the subset of rows that were not numeric to begin with.
        mask = ts_numeric.isna()
        if mask.any():
            # Use .copy() to avoid SettingWithCopyWarning
            parsed = parsed.copy()
            fallback = pd.to_datetime(ts_raw.loc[mask], utc=True, errors="coerce")
            try:
                parsed.loc[mask] = fallback
            except (TypeError, ValueError):
                # pandas >= 2.0: resolution mismatch (e.g., s vs us)
                parsed = parsed.astype(fallback.dtype)
                parsed.loc[mask] = fallback

        # Validate timestamp range to catch incorrect unit assumptions
        # (e.g., milliseconds instead of seconds)
        if not parsed.isna().all():
            min_ts = parsed.min()
            max_ts = parsed.max()
            # Reasonable range: 1970-01-01 to 2100-01-01
            reasonable_min = pd.Timestamp("1970-01-01", tz="UTC")
            reasonable_max = pd.Timestamp("2100-01-01", tz="UTC")
            if min_ts < reasonable_min or max_ts > reasonable_max:
                logger.warning(
                    f"Timestamp range [{min_ts}, {max_ts}] outside reasonable bounds. "
                    "Check that numeric timestamps are in seconds (not milliseconds/microseconds)."
                )

        df["timestamp"] = parsed

    # Check required columns after renaming
    required_cols = ["fUniqueID", "fBits", "timestamp", "adc", "fchannel"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}")

    # Convert numeric columns explicitly
    # Note: "adc" column existence already validated above
    adc_before = df["adc"].copy()
    df["adc"] = pd.to_numeric(df["adc"], errors="coerce")

    # Warn if any ADC values became NaN due to conversion
    num_invalid_adc = df["adc"].isna().sum() - adc_before.isna().sum()
    if num_invalid_adc > 0:
        logger.warning(f"Found {num_invalid_adc} non-numeric ADC values that were converted to NaN")

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

    # Sort by timestamp with secondary sort key for deterministic ordering
    # when multiple events have identical timestamps
    df = df.sort_values(["timestamp", "fUniqueID"]).reset_index(drop=True)

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

            # Validate timestamps are finite before converting to int
            if not (np.isfinite(t_min) and np.isfinite(t_max)):
                logger.warning("Timestamps contain NaN or infinity, skipping burst filter")
                return out_df, removed_total

            t_min_int = int(np.floor(t_min))
            t_max_int = int(np.ceil(t_max))
            hist, edges = np.histogram(times, bins=np.arange(t_min_int, t_max_int + 2))
            hist = hist.astype(int)

            win = int(micro_win)
            thr = int(micro_thr)
            if win > 0:
                csum = np.concatenate([[0], np.cumsum(hist)])
                counts = csum[win:] - csum[:-win]
                burst_bins = np.zeros_like(hist, dtype=bool)
                for i, c in enumerate(counts):
                    if c >= thr:
                        # Clip slice to array bounds
                        end_idx = min(i + win, len(burst_bins))
                        burst_bins[i:end_idx] = True
            else:
                burst_bins = np.zeros_like(hist, dtype=bool)

            bin_idx = np.searchsorted(edges, times, side="right") - 1
            # Clip indices to valid range [0, len(burst_bins)-1]
            bin_idx = np.clip(bin_idx, 0, len(burst_bins) - 1)
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
            and float(win) > 0
        ):
            t0 = times_sec.min()
            bins = pd.Series(
                ((times_sec - t0) // float(win)).astype(int), index=out_df.index
            )

            counts = bins.value_counts().sort_index()
            if counts.empty:
                full_index = range(0)
            else:
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
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results_folder = output_path / timestamp
        if results_folder.exists():
            raise FileExistsError(f"Results folder already exists: {results_folder}")
        results_folder.mkdir(parents=True, exist_ok=False)

    summary_path = results_folder / "summary.json"

    sanitized = _with_summary_key_aliases(to_native(summary_dict))

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
        try:
            sanitized = _canonicalize_config_keys(to_native(load_config(config_path)))
        except Exception:
            shutil.copyfile(Path(config_path), dest_path)
            logger.info(f"Copied config {config_path} -> {dest_path}")
        else:
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(sanitized, f, indent=4)
            logger.info(f"Wrote canonical config to {dest_path}")
    else:
        sanitized = _canonicalize_config_keys(to_native(config_path))
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=4)
        logger.info(f"Wrote config to {dest_path}")
    return dest_path
