# io_utils.py
import os
import shutil
import json
import logging
from datetime import datetime
import pandas as pd

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


def write_summary(output_dir, summary_dict):
    """
    Write out summary_dict to JSON in:
      <output_dir>/<timestamp_folder>/summary.json
    and returns path to that folder.
    """
    # Create timestamped subfolder
    now_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_folder = os.path.join(output_dir, now_str)
    ensure_dir(results_folder)

    summary_path = os.path.join(results_folder, "summary.json")

    # Convert numpy types to native Python
    def convert(o):
        if isinstance(o, (int, float, str, bool, list, dict)) or o is None:
            return o
        try:
            return o.item()
        except Exception:
            return str(o)

    # Recursively convert all items
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        else:
            return convert(obj)

    sanitized = sanitize(summary_dict)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=4)

    logger.info(f"Wrote summary JSON to {summary_path}")
    return results_folder


def copy_config(results_dir, config_path):
    """
    Copy the used config JSON into the provided results directory.
    Expects ``results_dir`` to be the folder returned by :func:`write_summary`.
    Returns destination path.
    """
    if not os.path.isdir(results_dir):
        raise RuntimeError(
            f"Results directory '{results_dir}' does not exist."
        )

    dest_path = os.path.join(results_dir, "config_used.json")
    shutil.copyfile(config_path, dest_path)
    logger.info(f"Copied config {config_path} -> {dest_path}")
    return dest_path
