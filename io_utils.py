# io_utils.py
import os
import shutil
import json
import logging
import builtins
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# Provide JSON-like boolean/null names for tests that use them directly
builtins.true = True
builtins.false = False
builtins.null = None


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

    # Basic validation: ensure we decoded a dictionary
    if not isinstance(cfg, dict):
        raise TypeError("Config JSON must decode to a dictionary")

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


def load_data(csv_path):
    """Convenience wrapper returning timestamp and adc arrays."""
    df = load_events(csv_path)
    return df["timestamp"].to_numpy(), df["adc"].to_numpy()


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


def copy_config(output_dir, config_path):
    """
    Copy the used config JSON into the same timestamped results folder.
    Must be called *after* write_summary(), so that the timestamped folder exists.
    Returns destination path.
    """
    # Identify the single subfolder in output_dir (should be timestamped)   assume only one new one
    subfolders = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]
    if not subfolders:
        raise RuntimeError(
            f"No subfolders found in {output_dir} to copy config into."
        )
    # Pick the folder with the lexicographically largest name (most recent timestamp)
    timestamped = sorted(subfolders)[-1]
    dest_folder = os.path.join(output_dir, timestamped)

    dest_path = os.path.join(dest_folder, "config_used.json")
    shutil.copyfile(config_path, dest_path)
    logger.info(f"Copied config {config_path}   {dest_path}")
    return dest_path
