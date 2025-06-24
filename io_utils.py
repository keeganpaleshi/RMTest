# io_utils.py
from pathlib import Path
import shutil
import json
import logging
import warnings
from datetime import datetime, timezone
import pandas as pd
from constants import load_nuclide_overrides

import numpy as np
from utils import to_native, parse_datetime
import jsonschema


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
        out[iso] = events.loc[mask, "timestamp"].values.astype(float)
    return out

logger = logging.getLogger(__name__)


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
        "spectral_fit": {
            "type": "object",
            "properties": {"expected_peaks": {"type": "object"}},
        },
        "time_fit": {
            "type": "object",
            "properties": {
                "do_time_fit": {"type": "boolean"},
                "hl_po214": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 2,
                },
                "hl_po218": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 2,
                },
                "hl_po210": {
                    "type": "array",
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
        "baseline": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "range": {"type": "array", "items": {"type": ["string", "number"]}, "minItems": 2, "maxItems": 2},
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
                "init_sigma_adc": {"type": "number", "minimum": 0},
                "init_tau_adc": {"type": "number", "minimum": 0},
                "sanity_tolerance_mev": {"type": "number", "minimum": 0},
                "known_energies": {"type": "object"},
            },
        },
        "analysis": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "analysis_start_time": {"type": ["string", "number", "null"]},
                "analysis_end_time": {"type": ["string", "number", "null"]},
                "spike_end_time": {"type": ["string", "number", "null"]},
                "spike_periods": {
                    "type": ["array", "null"],
                    "items": {"type": "array", "items": {"type": ["string", "number"]}, "minItems": 2, "maxItems": 2},
                },
                "run_periods": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": ["string", "number"]}, "minItems": 2, "maxItems": 2},
                },
                "radon_interval": {"type": "array", "items": {"type": ["string", "number"]}, "minItems": 2, "maxItems": 2},
                "ambient_concentration": {"type": ["number", "null"]},
                "settle_s": {"type": ["number", "null"], "minimum": 0},
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


def _no_duplicates_object_pairs_hook(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError(f"Duplicate key '{k}' in configuration")
        obj[k] = v
    return obj


def ensure_dir(path):
    """Create directory if it does not exist."""
    p = Path(path)
    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)



def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` and return ``base``."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path):
    """
    Load ``config_path`` and merge it with ``config_defaults.json`` if present.
    The resulting dictionary is validated against ``CONFIG_SCHEMA``.

    Returns the merged configuration dictionary.
    Raises ``FileNotFoundError`` or ``json.JSONDecodeError`` on failure.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    defaults_path = Path(__file__).resolve().with_name("config_defaults.json")
    defaults: dict = {}
    if defaults_path.is_file():
        with open(defaults_path, "r", encoding="utf-8") as f:
            defaults = json.load(f, object_pairs_hook=_no_duplicates_object_pairs_hook)

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f, object_pairs_hook=_no_duplicates_object_pairs_hook)

    if defaults:
        cfg = _merge_dicts(defaults, cfg)


    try:
        jsonschema.validate(cfg, CONFIG_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        if e.validator == "required":
            raise KeyError(e.message)
        raise

    cfg["nuclide_constants"] = load_nuclide_overrides(cfg)

    # CONFIG_SCHEMA validation already checks required keys

    return cfg


def load_events(csv_path, *, column_map=None):
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

    # Parse timestamps (epoch seconds) into timezone-aware datetimes
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")

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

    logger.info(
        f"Loaded {len(df)} events from {csv_path} ({discarded} discarded)."
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
        converted to ``datetime64`` using :func:`parse_datetime` and the
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
        ts = ts.map(parse_datetime)
        out_df["timestamp"] = ts
    times_sec = ts.view("int64").to_numpy() / 1e9

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
                ts = ts.map(parse_datetime)
                out_df["timestamp"] = ts
            times_sec = ts.view("int64").to_numpy() / 1e9

    # ───── rate-based veto ─────
    if mode in ("rate", "both"):
        win = bcfg.get("burst_window_size_s")
        roll = bcfg.get("rolling_median_window")
        mult = bcfg.get("burst_multiplier")

        if win is not None and roll is not None and mult is not None and len(out_df) > 0:
            t0 = times_sec.min()
            bins = pd.Series(((times_sec - t0) // float(win)).astype(int), index=out_df.index)

            counts = bins.value_counts().sort_index()
            full_index = range(counts.index.min(), counts.index.max() + 1)
            counts_full = counts.reindex(full_index, fill_value=0)

            med = (
                counts_full
                .rolling(int(roll), center=True, min_periods=1)
                .median()
            )

            threshold = mult * med
            burst_bins = (
                counts_full.astype(float)[counts_full.astype(float) > threshold]
                .index.to_numpy(dtype=np.int64)
            )
            mask = ~bins.isin(burst_bins)

            removed_total += int((~mask).sum())
            out_df = out_df[mask].reset_index(drop=True)

    return out_df, removed_total


def write_summary(output_dir, summary_dict, timestamp=None):
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
