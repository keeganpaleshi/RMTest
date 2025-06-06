# io_utils.py
import os
import shutil
import json
import logging
from datetime import datetime
import pandas as pd

import numpy as np
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

    # Basic validation: check for required keys within each section
    required_structure = {
        "pipeline": ["log_level"],
        "spectral_fit": ["expected_peaks"],
        "time_fit": ["do_time_fit"],
        "systematics": ["enable"],
        "plotting": ["plot_save_formats"],
    }

    for section, keys in required_structure.items():
        if section not in cfg:
            raise KeyError(f"Missing required config section: '{section}'")
        for key in keys:
            if key not in cfg.get(section, {}):
                raise KeyError(f"Missing key '{section}.{key}' in config")

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

    # Drop rows with non-finite timestamp or adc values
    start_len = len(df)
    mask = np.isfinite(df["timestamp"]) & np.isfinite(df["adc"])
    df = df[mask]

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    discarded = start_len - len(df)

    # Convert types
    df["timestamp"] = df["timestamp"].astype(int)
    df["adc"] = df["adc"].astype(int)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} events from {csv_path} ({discarded} discarded)."
    )
    return df


def apply_burst_filter(df, cfg, mode="rate"):
    """Remove events occurring during high-rate bursts.

    ``mode`` selects the filtering strategy:

    ``'none'`` – return the input unchanged;
    ``'micro'`` – short sliding-window burst veto;
    ``'rate'`` – rolling-median rate-based veto (legacy behaviour);
    ``'both'`` – apply the micro window first then the rate veto.

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

    if mode == "none" or len(df) == 0:
        return df.copy(), 0

    removed_total = 0
    out_df = df.copy()

    # ───── micro-burst veto ─────
    if mode in ("micro", "both"):
        micro_win = bcfg.get("micro_window_size_s")
        micro_thr = bcfg.get("micro_count_threshold")

        if micro_win is not None and micro_thr is not None:
            times = out_df["timestamp"].values.astype(float)
            window_end = times + float(micro_win)
            j = np.searchsorted(times, window_end, side="right")
            counts = j - np.arange(len(times))
            starts = np.nonzero(counts >= int(micro_thr))[0]

            if len(starts) > 0:
                diff = np.zeros(len(times) + 1, dtype=int)
                diff[starts] += 1
                diff[j[starts]] -= 1
                to_remove = np.cumsum(diff[:-1]) > 0
            else:
                to_remove = np.zeros(len(times), dtype=bool)

            removed_total += int(to_remove.sum())
            out_df = out_df[~to_remove].reset_index(drop=True)

    # ───── rate-based veto ─────
    if mode in ("rate", "both"):
        win = bcfg.get("burst_window_size_s")
        roll = bcfg.get("rolling_median_window")
        mult = bcfg.get("burst_multiplier")

        if win is not None and roll is not None and mult is not None and len(out_df) > 0:
            t0 = out_df["timestamp"].min()
            bins = ((out_df["timestamp"] - t0) // float(win)).astype(int)

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

            removed_total += int((~mask).sum())
            out_df = out_df[mask].reset_index(drop=True)

    return out_df, removed_total


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
