"""Command-line argument parser for the Radon Monitor Analysis Pipeline."""

import argparse
from pathlib import Path


def parse_args(argv=None):
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Full Radon Monitor Analysis Pipeline",
        epilog=(
            "See the README's Opt-in section for details on experimental flags "
            "such as background_model=loglin_unit and likelihood=extended."
        ),
    )
    default_cfg = Path(__file__).resolve().with_name("config.yaml")
    p.add_argument(
        "--config",
        "-c",
        default=str(default_cfg),
        help="Path to YAML or JSON configuration file (default: config.yaml)",
    )
    default_input = Path.cwd() / "merged_output.csv"
    p.add_argument(
        "--input",
        "-i",
        default=str(default_input),
        help=(
            "CSV of merged event data (must contain at least: timestamp, adc) "
            f"(default: {default_input})"
        ),
    )
    p.add_argument(
        "--output_dir",
        "-o",
        default="results",
        help=(
            "Directory under which to create a timestamped analysis folder "
            "(override with --job-id; default: results)"
        ),
    )
    p.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone for naive input timestamps (default: UTC)",
    )
    p.add_argument(
        "--baseline_range",
        nargs=2,
        metavar=("TSTART", "TEND"),
        type=str,
        help=(
            "Optional baseline-run interval. Providing this option overrides `baseline.range` in config.yaml. Provide two values (either ISO strings or epoch floats). If set, those events are extracted (same energy cuts) and listed in `baseline` of the summary."
        ),
    )
    p.add_argument(
        "--baseline-mode",
        choices=["none", "electronics", "radon", "all"],
        default="all",
        help="Background removal strategy (default: all)",
    )
    p.add_argument(
        "--iso",
        choices=["radon", "po218", "po214"],
        help=(
            "Select which progeny drives the radon estimate "
            "(overrides analysis_isotope in config.yaml)"
        ),
    )
    p.add_argument(
        "--allow-negative-baseline",
        action="store_true",
        help="Allow negative baseline-corrected rates",
    )
    p.add_argument(
        "--allow-negative-activity",
        action="store_true",
        help="Continue if radon activity is negative",
    )
    p.add_argument(
        "--check-baseline-only",
        action="store_true",
        help="Exit after printing baseline diagnostics",
    )
    p.add_argument(
        "--burst-mode",
        choices=["none", "micro", "rate", "both"],
        help="Burst filtering mode to pass to apply_burst_filter. Providing this option overrides `burst_filter.burst_mode` in config.yaml",
    )
    p.add_argument(
        "--burst-sensitivity-scan",
        action="store_true",
        help="Scan burst parameters and plot activity vs burst window/multiplier",
    )
    p.add_argument(
        "--job-id",
        help="Optional identifier used for the results folder instead of the timestamp",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results folder if it already exists",
    )
    p.add_argument(
        "--efficiency-json",
        help="Path to a JSON file containing efficiency inputs to merge into the configuration",
    )
    p.add_argument(
        "--systematics-json",
        help="Path to a JSON file with systematics settings overriding the config",
    )
    p.add_argument(
        "--spike-count",
        type=float,
        help="Counts observed during a spike run for efficiency",
    )
    p.add_argument(
        "--spike-count-err",
        type=float,
        help="Uncertainty on spike counts",
    )
    p.add_argument(
        "--spike-activity",
        type=float,
        help="Known spike activity in Bq",
    )
    p.add_argument(
        "--spike-duration",
        type=float,
        help="Duration of the spike run in seconds",
    )
    p.add_argument(
        "--no-spike",
        action="store_true",
        help="Disable spike efficiency",
    )
    p.add_argument(
        "--analysis-end-time",
        type=str,
        help="Ignore events occurring after this ISO timestamp. Providing this option overrides `analysis.analysis_end_time` in config.yaml",
    )
    p.add_argument(
        "--analysis-start-time",
        type=str,
        help="Reference start time of the analysis (ISO string or epoch). Overrides `analysis.analysis_start_time` in config.yaml",
    )
    p.add_argument(
        "--background-model",
        choices=["linear", "loglin_unit"],
        help="Experimental (opt-in) background model. Defaults keep legacy behavior.",
    )
    p.add_argument(
        "--likelihood",
        choices=["current", "extended"],
        help="Experimental (opt-in) likelihood. Defaults keep legacy behavior.",
    )
    p.add_argument(
        "--spike-start-time",
        help="Discard events after this ISO timestamp. Providing this option overrides `analysis.spike_start_time` in config.yaml",
    )
    p.add_argument(
        "--spike-end-time",
        help="Discard events before this ISO timestamp. Providing this option overrides `analysis.spike_end_time` in config.yaml",
    )
    p.add_argument(
        "--spike-period",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Discard events between START and END (can be given multiple times). Providing this option overrides `analysis.spike_periods` in config.yaml",
    )
    p.add_argument(
        "--run-period",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Keep events between START and END (can be given multiple times). Providing this option overrides `analysis.run_periods` in config.yaml",
    )
    p.add_argument(
        "--radon-interval",
        nargs=2,
        metavar=("START", "END"),
        help="Time interval to evaluate radon delta. Providing this option overrides `analysis.radon_interval` in config.yaml",
    )
    p.add_argument(
        "--slope",
        type=float,
        help="Apply a linear ADC drift correction with the given slope. Providing this option overrides `systematics.adc_drift_rate` in config.yaml",
    )
    p.add_argument(
        "--noise-cutoff",
        type=int,
        help=(
            "ADC threshold for the noise cut. Providing this option overrides "
            "`calibration.noise_cutoff` in config.yaml"
        ),
    )
    p.add_argument(
        "--calibration-slope",
        type=float,
        help=(
            "Fixed MeV per ADC conversion slope. Providing this option overrides "
            "`calibration.slope_MeV_per_ch` in config.yaml"
        ),
    )
    p.add_argument(
        "--float-slope",
        action="store_true",
        help="Allow provided calibration slope to float during the two-point fit",
    )
    p.add_argument(
        "--calibration-method",
        choices=["two-point", "auto"],
        help=(
            "Energy calibration method. Providing this option overrides "
            "`calibration.method` in config.yaml"
        ),
    )
    p.add_argument(
        "--settle-s",
        type=float,
        help="Discard events occurring this many seconds after the start",
    )
    p.add_argument(
        "--hl-po214",
        type=float,
        help=(
            "Half-life to use for Po-214 in seconds. "
            "Providing this option overrides `time_fit.hl_po214` in config.yaml"
        ),
    )
    p.add_argument(
        "--hl-po218",
        type=float,
        help=(
            "Half-life to use for Po-218 in seconds. "
            "Providing this option overrides `time_fit.hl_po218` in config.yaml"
        ),
    )
    p.add_argument(
        "--eff-fixed",
        action="store_true",
        help="Fix all efficiencies to exactly 1.0 (no prior)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging. Providing this option overrides `pipeline.log_level` in config.yaml",
    )
    p.add_argument(
        "--plot-time-binning-mode",
        dest="time_bin_mode_new",
        choices=["auto", "fd", "fixed"],
        help="Time-series binning mode. Providing this option overrides `plotting.plot_time_binning_mode` in config.yaml",
    )
    p.add_argument(
        "--time-bin-mode",
        dest="time_bin_mode_old",
        choices=["auto", "fd", "fixed"],
        help="DEPRECATED alias for --plot-time-binning-mode",
    )
    p.add_argument(
        "--plot-time-bin-width",
        dest="time_bin_width",
        type=float,
        help="Fixed time bin width in seconds. Providing this option overrides `plotting.plot_time_bin_width_s` in config.yaml",
    )
    p.add_argument(
        "--dump-ts-json",
        "--dump-time-series-json",
        dest="dump_ts_json",
        action="store_true",
        help="Write *_ts.json files containing binned time-series data",
    )
    p.add_argument(
        "--ambient-file",
        help=(
            "Two-column text file of timestamp and ambient concentration in Bq/L"
        ),
    )
    p.add_argument(
        "--ambient-concentration",
        type=float,
        help=(
            "Ambient radon concentration in Bq per liter for "
            "equivalent air plot. Providing this option overrides "
            "`analysis.ambient_concentration` in config.yaml"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        help="Override random seed used by analysis algorithms. Providing this option overrides `pipeline.random_seed` in config.yaml",
    )
    p.add_argument(
        "--palette",
        help="Color palette for plots. Providing this option overrides `plotting.palette` in config.yaml",
    )
    p.add_argument(
        "--strict-covariance",
        action="store_true",
        help="Fail if fit covariance matrices are not positive definite",
    )
    p.add_argument(
        "--hierarchical-summary",
        metavar="OUTFILE",
        help=(
            "Combine half-life and calibration results from previous runs and "
            "write a hierarchical fit summary to OUTFILE"
        ),
    )
    p.add_argument(
        "--reproduce",
        metavar="SUMMARY",
        help="Load config and seed from SUMMARY to reproduce a previous run",
    )

    args = p.parse_args(argv)

    if args.time_bin_mode_new is not None and args.time_bin_mode_old is not None:
        if args.time_bin_mode_new != args.time_bin_mode_old:
            p.error(
                "--plot-time-binning-mode conflicts with deprecated --time-bin-mode"
            )

    args.time_bin_mode = (
        args.time_bin_mode_new
        if args.time_bin_mode_new is not None
        else args.time_bin_mode_old
    )
    del args.time_bin_mode_new
    del args.time_bin_mode_old

    return args
