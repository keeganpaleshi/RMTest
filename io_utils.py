# io_utils.py
import os
import shutil
import json
import logging
from datetime import datetime
import pandas as pd

from utils import to_native

logger = logging.getLogger(__name__)


def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_config(config_path):
    """
    Load and validate JSON configuration.
    Returns a dict.
    Raises FileNotFoundError or json.JSONDecodeError on failure.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Basic validation: check for required top level keys
    required_sections = [
        "pipeline",
        "spectral_fit",
        "time_fit",
        "systematics",
        "plotting",
    ]
    for section in required_sections:
        if section not in cfg:
            raise KeyError(f"Missing required config section: '{section}'")

    return cfg


def load_events(csv_path):
    """
    Read event CSV into a DataFrame with columns:
       ['fUniqueID','fBits','timestamp','adc','fchannel']
    - Ensures timestamp is int, adc is int.
    - Sort ascending by timestamp.
    - Returns DataFrame.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Read CSV; assume no header comments, first row is header
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ["fUniqueID", "fBits", "timestamp", "adc", "fchannel"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}")

    # Convert types
    df["timestamp"] = df["timestamp"].astype(int)
    df["adc"] = df["adc"].astype(int)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} events from {csv_path}.")
    return df


def apply_burst_filter(df, cfg):
    """Remove events occurring during high-rate bursts.

    Parameters
    ----------
    df : pandas.DataFrame
        Event data containing a ``timestamp`` column in seconds.
    cfg : dict
        Configuration dictionary.  Expected keys under ``burst_filter`` are
        ``burst_window_size_s``, ``rolling_median_window`` and
        ``burst_multiplier``.

    Returns
    -------
    DataFrame
        Filtered events with bursts removed.
    int
        Number of events removed.
    """

    bcfg = cfg.get("burst_filter", {})
    win = bcfg.get("burst_window_size_s")
    roll = bcfg.get("rolling_median_window")
    mult = bcfg.get("burst_multiplier")

    if win is None or roll is None or mult is None:
        return df.copy(), 0

    if len(df) == 0:
        return df.copy(), 0

    # Bin indices relative to first timestamp
    t0 = df["timestamp"].min()
    bins = ((df["timestamp"] - t0) // float(win)).astype(int)

    counts = bins.value_counts().sort_index()
    full_index = range(counts.index.min(), counts.index.max() + 1)
    counts_full = counts.reindex(full_index, fill_value=0)

    med = (
        counts_full
        .rolling(int(roll), center=True, min_periods=1)
        .median()
    )

    threshold = mult * med
    burst_bins = counts_full[counts_full > threshold].index
    mask = ~bins.isin(burst_bins)

    removed = int((~mask).sum())
    return df[mask].reset_index(drop=True), removed


def write_summary(output_dir, summary_dict, timestamp=None):
    """
    Write out ``summary_dict`` to ``summary.json`` under a timestamped
    directory and return the path to that directory.

    Parameters
    ----------
    output_dir : str
        Parent directory under which the results folder will be created.
    summary_dict : dict
        Data to serialise into ``summary.json``.
    timestamp : str, optional
        Timestamp string to use for the results folder.  If ``None`` a new
        UTC timestamp will be generated.
    """
    # Create timestamped subfolder
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_folder = os.path.join(output_dir, timestamp)
    ensure_dir(results_folder)

    summary_path = os.path.join(results_folder, "summary.json")

    # Convert numpy types to native Python using shared helper
    sanitized = to_native(summary_dict)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=4)

    logger.info(f"Wrote summary JSON to {summary_path}")
    return results_folder


def copy_config(output_dir, config_path):
    """
    Copy the used config JSON into the timestamped results folder.

    ``output_dir`` can either be the parent directory passed to
    :func:`write_summary` or the timestamped directory returned by it.
    In both cases the configuration file will be copied alongside the
    generated ``summary.json``.

    Returns the destination path.
    """

    # If ``summary.json`` exists directly under ``output_dir`` we assume the
    # caller provided the timestamped folder path.
    if os.path.isfile(os.path.join(output_dir, "summary.json")):
        dest_folder = output_dir
    else:
        subfolders = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]
        if not subfolders:
            raise RuntimeError(
                f"No subfolders found in {output_dir} to copy config into."
            )
        # Pick the most recent (lexicographically largest) folder
        timestamped = sorted(subfolders)[-1]
        dest_folder = os.path.join(output_dir, timestamped)

    dest_path = os.path.join(dest_folder, "config_used.json")
    shutil.copyfile(config_path, dest_path)
    logger.info(f"Copied config {config_path} -> {dest_path}")
    return dest_path
