"""
pipeline_init.py

Pipeline initialization and configuration setup for analyze.py.

This module handles:
- Git commit and requirements hashing
- CLI argument parsing and path conversion
- Timezone resolution and time argument parsing
- Configuration loading and CLI overrides
- Logging setup
- Random seed initialization
"""

import sys
import logging
import random
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dateutil.tz import gettz

from cli_parser import parse_args
from io_utils import load_config
from utils import parse_time_arg, to_native
from analysis_helpers import PipelineTimer, _safe_float
from reporting import start_warning_capture


def get_git_commit():
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8"
        ).strip()
    except Exception:
        return "unknown"


def get_requirements_hash():
    """Get the SHA256 hash of requirements.txt, or 'unknown' if not found."""
    try:
        req_path = Path(__file__).resolve().parent / "requirements.txt"
        with open(req_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "unknown"


def compute_cli_hash(argv=None):
    """Compute SHA256 hash of the CLI arguments."""
    cli_args = [sys.argv[0]] + (list(argv) if argv is not None else sys.argv[1:])
    return hashlib.sha256(" ".join(cli_args).encode("utf-8")).hexdigest()


def convert_args_to_paths(args):
    """Convert CLI argument paths to Path objects."""
    args.config = Path(args.config)
    args.input = Path(args.input)
    args.output_dir = Path(args.output_dir)
    if args.efficiency_json:
        args.efficiency_json = Path(args.efficiency_json)
    if args.systematics_json:
        args.systematics_json = Path(args.systematics_json)
    if args.ambient_file:
        args.ambient_file = Path(args.ambient_file)
    if args.hierarchical_summary:
        args.hierarchical_summary = Path(args.hierarchical_summary)
    return args


def resolve_timezone(args):
    """Resolve the timezone from args and validate it."""
    tzinfo = gettz(args.timezone)
    if tzinfo is None:
        logging.error("Unknown timezone '%s'", args.timezone)
        sys.exit(1)
    return tzinfo


def parse_time_arguments(args, tzinfo):
    """Parse all time-related arguments using the given timezone."""
    if args.baseline_range:
        args.baseline_range = [
            parse_time_arg(t, tz=tzinfo) for t in args.baseline_range
        ]
    if args.analysis_end_time is not None:
        args.analysis_end_time = parse_time_arg(args.analysis_end_time, tz=tzinfo)
    if args.analysis_start_time is not None:
        args.analysis_start_time = parse_time_arg(args.analysis_start_time, tz=tzinfo)
    if args.spike_start_time is not None:
        args.spike_start_time = parse_time_arg(args.spike_start_time, tz=tzinfo)
    if args.spike_end_time is not None:
        args.spike_end_time = parse_time_arg(args.spike_end_time, tz=tzinfo)
    if args.spike_period:
        args.spike_period = [
            [parse_time_arg(s, tz=tzinfo), parse_time_arg(e, tz=tzinfo)]
            for s, e in args.spike_period
        ]
    if args.run_period:
        args.run_period = [
            [parse_time_arg(s, tz=tzinfo), parse_time_arg(e, tz=tzinfo)]
            for s, e in args.run_period
        ]
    if args.radon_interval:
        args.radon_interval = [
            parse_time_arg(args.radon_interval[0], tz=tzinfo),
            parse_time_arg(args.radon_interval[1], tz=tzinfo),
        ]
    return args


def handle_reproduce_mode(args):
    """Handle reproducibility mode if specified."""
    if args.reproduce:
        rep_path = Path(args.reproduce)
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                rep_summary = json.load(f)
        except Exception as e:
            logging.error("Could not load summary '%s': %s", args.reproduce, e)
            sys.exit(1)
        args.config = rep_path.parent / "config_used.json"
        args.seed = rep_summary.get("random_seed")
    return args


def apply_config_overrides(cfg, args):
    """Apply all CLI argument overrides to the configuration."""

    def _log_override(section, key, new_val):
        prev = cfg.get(section, {}).get(key)
        if prev is not None and prev != new_val:
            logging.info(
                f"Overriding {section}.{key}={prev!r} with {new_val!r} from CLI"
            )

    # Load external JSON files
    if args.efficiency_json:
        try:
            with open(args.efficiency_json, "r", encoding="utf-8") as f:
                cfg["efficiency"] = json.load(f)
        except Exception as e:
            logging.error(
                "Could not load efficiency JSON '%s': %s", args.efficiency_json, e
            )
            sys.exit(1)

    if args.systematics_json:
        try:
            with open(args.systematics_json, "r", encoding="utf-8") as f:
                cfg["systematics"] = json.load(f)
        except Exception as e:
            logging.error(
                "Could not load systematics JSON '%s': %s",
                args.systematics_json,
                e,
            )
            sys.exit(1)

    # Pipeline settings
    if args.seed is not None:
        _log_override("pipeline", "random_seed", int(args.seed))
        cfg.setdefault("pipeline", {})["random_seed"] = int(args.seed)

    if args.debug:
        cfg.setdefault("pipeline", {})["log_level"] = "DEBUG"

    # Analysis settings
    if args.ambient_concentration is not None:
        ambient_cli = _safe_float(args.ambient_concentration)
        if ambient_cli is None:
            logging.warning(
                "Ignoring ambient concentration override %r; could not convert to float",
                args.ambient_concentration,
            )
        else:
            _log_override(
                "analysis",
                "ambient_concentration",
                ambient_cli,
            )
            cfg.setdefault("analysis", {})["ambient_concentration"] = ambient_cli

    if args.analysis_end_time is not None:
        _log_override("analysis", "analysis_end_time", args.analysis_end_time)
        cfg.setdefault("analysis", {})["analysis_end_time"] = args.analysis_end_time

    if args.analysis_start_time is not None:
        _log_override("analysis", "analysis_start_time", args.analysis_start_time)
        cfg.setdefault("analysis", {})["analysis_start_time"] = args.analysis_start_time

    if args.spike_start_time is not None:
        _log_override("analysis", "spike_start_time", args.spike_start_time)
        cfg.setdefault("analysis", {})["spike_start_time"] = args.spike_start_time

    if args.spike_end_time is not None:
        _log_override("analysis", "spike_end_time", args.spike_end_time)
        cfg.setdefault("analysis", {})["spike_end_time"] = args.spike_end_time

    if args.spike_period:
        _log_override("analysis", "spike_periods", args.spike_period)
        cfg.setdefault("analysis", {})["spike_periods"] = [
            [s, e] for s, e in args.spike_period
        ]

    if args.run_period:
        _log_override("analysis", "run_periods", args.run_period)
        cfg.setdefault("analysis", {})["run_periods"] = [
            [s, e] for s, e in args.run_period
        ]

    if args.radon_interval:
        _log_override("analysis", "radon_interval", args.radon_interval)
        cfg.setdefault("analysis", {})["radon_interval"] = [
            args.radon_interval[0],
            args.radon_interval[1],
        ]

    if args.background_model is not None:
        _log_override("analysis", "background_model", args.background_model)
        cfg.setdefault("analysis", {})["background_model"] = args.background_model

    if args.likelihood is not None:
        _log_override("analysis", "likelihood", args.likelihood)
        cfg.setdefault("analysis", {})["likelihood"] = args.likelihood

    if args.settle_s is not None:
        _log_override("analysis", "settle_s", float(args.settle_s))
        cfg.setdefault("analysis", {})["settle_s"] = float(args.settle_s)

    # Time fit settings
    if args.hl_po214 is not None:
        tf = cfg.setdefault("time_fit", {})
        sig = 0.0
        current = tf.get("hl_po214")
        if isinstance(current, list) and len(current) > 1:
            sig = current[1]
        _log_override("time_fit", "hl_po214", [float(args.hl_po214), sig])
        tf["hl_po214"] = [float(args.hl_po214), sig]

    if args.hl_po218 is not None:
        tf = cfg.setdefault("time_fit", {})
        sig = 0.0
        current = tf.get("hl_po218")
        if isinstance(current, list) and len(current) > 1:
            sig = current[1]
        _log_override("time_fit", "hl_po218", [float(args.hl_po218), sig])
        tf["hl_po218"] = [float(args.hl_po218), sig]

    # Plotting settings
    if args.time_bin_mode:
        _log_override("plotting", "plot_time_binning_mode", args.time_bin_mode)
        cfg.setdefault("plotting", {})["plot_time_binning_mode"] = args.time_bin_mode

    if args.time_bin_width is not None:
        _log_override(
            "plotting",
            "plot_time_bin_width_s",
            float(args.time_bin_width),
        )
        cfg.setdefault("plotting", {})["plot_time_bin_width_s"] = float(
            args.time_bin_width
        )

    if args.dump_ts_json:
        cfg.setdefault("plotting", {})["dump_time_series_json"] = True

    if args.palette:
        cfg.setdefault("plotting", {})["palette"] = args.palette

    # Burst filter settings
    if args.burst_mode is not None:
        _log_override("burst_filter", "burst_mode", args.burst_mode)
        cfg.setdefault("burst_filter", {})["burst_mode"] = args.burst_mode

    # Efficiency spike settings
    if (
        args.spike_count is not None
        or args.spike_count_err is not None
        or args.spike_activity is not None
        or args.spike_duration is not None
        or args.no_spike
    ):
        eff_sec = cfg.setdefault("efficiency", {}).setdefault("spike", {})
        if args.spike_count is not None:
            eff_sec["counts"] = float(args.spike_count)
        if args.spike_count_err is not None:
            eff_sec["error"] = float(args.spike_count_err)
        if args.spike_activity is not None:
            eff_sec["activity_bq"] = float(args.spike_activity)
        if args.spike_duration is not None:
            eff_sec["live_time_s"] = float(args.spike_duration)
        if args.no_spike:
            eff_sec["enabled"] = False

    # Systematics settings
    if args.slope is not None:
        _log_override("systematics", "adc_drift_rate", float(args.slope))
        cfg.setdefault("systematics", {})["adc_drift_rate"] = float(args.slope)

    # Calibration settings
    if args.noise_cutoff is not None:
        _log_override(
            "calibration",
            "noise_cutoff",
            int(args.noise_cutoff),
        )
        cfg.setdefault("calibration", {})["noise_cutoff"] = int(args.noise_cutoff)

    if args.calibration_slope is not None:
        _log_override(
            "calibration",
            "slope_MeV_per_ch",
            float(args.calibration_slope),
        )
        cfg.setdefault("calibration", {})["slope_MeV_per_ch"] = float(args.calibration_slope)

    if args.float_slope:
        cfg.setdefault("calibration", {})["float_slope"] = True

    if args.calibration_method is not None:
        _log_override("calibration", "method", args.calibration_method)
        cfg.setdefault("calibration", {})["method"] = args.calibration_method

    # Analysis isotope
    if args.iso is not None:
        prev = cfg.get("analysis_isotope")
        if prev is not None and prev != args.iso:
            logging.info(
                f"Overriding analysis_isotope={prev!r} with {args.iso!r} from CLI"
            )
        cfg["analysis_isotope"] = args.iso

    assert cfg.get("analysis_isotope", "radon") in {"radon", "po218", "po214"}

    # Baseline settings
    if args.allow_negative_baseline:
        cfg["allow_negative_baseline"] = True

    return cfg


def setup_logging(cfg):
    """Configure logging based on config settings."""
    log_level = cfg.get("pipeline", {}).get("log_level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level, format="%(levelname)s:%(name)s:%(message)s"
    )
    start_warning_capture()


def setup_random_seed(cfg, now_str, base_cfg_sha):
    """Setup random seed and return the seed used."""
    import numpy as np

    seed = cfg.get("pipeline", {}).get("random_seed")
    seed_used = None

    if seed is not None:
        try:
            seed_int = int(seed)
            np.random.seed(seed_int)
            random.seed(seed_int)
            seed_used = seed_int
        except Exception:
            logging.warning(f"Invalid random_seed '{seed}' ignored")
    else:
        derived_seed = int(
            hashlib.sha256((base_cfg_sha + now_str).encode("utf-8")).hexdigest()[:8],
            16,
        )
        np.random.seed(derived_seed)
        random.seed(derived_seed)
        seed_used = derived_seed
        cfg.setdefault("pipeline", {})["random_seed"] = derived_seed

    return seed_used


def init_pipeline(argv=None):
    """
    Initialize the pipeline with all necessary setup.

    Returns:
        tuple: (args, cfg, timer, tzinfo, metadata_dict)
            - args: Parsed CLI arguments
            - cfg: Configuration dictionary with overrides applied
            - timer: PipelineTimer instance
            - tzinfo: Timezone info object
            - metadata_dict: Dictionary with hashes and metadata
    """
    # Compute hashes
    cli_sha256 = compute_cli_hash(argv)
    commit = get_git_commit()
    requirements_sha256 = get_requirements_hash()

    # Parse arguments
    args = parse_args(argv)
    timer = PipelineTimer(logging.getLogger("analyze.timer"))

    # Handle reproduce mode
    args = handle_reproduce_mode(args)

    # Convert paths
    args = convert_args_to_paths(args)

    # Resolve timezone
    tzinfo = resolve_timezone(args)

    # Parse time arguments
    args = parse_time_arguments(args, tzinfo)

    # Load configuration
    with timer.section("load_config"):
        try:
            cfg = load_config(args.config)
        except Exception as e:
            logging.error("Could not load config '%s': %s", args.config, e)
            sys.exit(1)

    # Apply CLI overrides
    cfg = apply_config_overrides(cfg, args)

    # Generate timestamp
    now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Compute base config hash (before random seed)
    base_cfg_sha = hashlib.sha256(
        json.dumps(to_native(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Setup logging
    setup_logging(cfg)

    # Setup random seed
    seed_used = setup_random_seed(cfg, now_str, base_cfg_sha)

    # Compute final config hash
    cfg_sha256 = hashlib.sha256(
        json.dumps(to_native(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Collect metadata
    metadata = {
        "cli_sha256": cli_sha256,
        "commit": commit,
        "requirements_sha256": requirements_sha256,
        "cfg_sha256": cfg_sha256,
        "seed_used": seed_used,
        "timestamp": now_str,
    }

    return args, cfg, timer, tzinfo, metadata
