import os
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from io_utils import load_config, load_events, write_summary, copy_config


def test_load_config(tmp_path):
    cfg = {
        "adc": {
            "min_channel": 0,
            "max_channel": 1000,
            "peak_detection": {"height": 1, "distance": 1, "prominence": 1},
        },
        "efficiency": {"eff_po214": 0.4, "eff_po218": 0.8, "eff_override_all": 0.0},
        "burst_filter": {"window_seconds": 60, "threshold_factor": 10},
        "time_series": {"time_bin_seconds": 3600},
        "fit_options": {
            "fit_po214_only": True,
            "fit_po218_po214": False,
            "fix_B": False,
            "fix_N0": False,
            "baseline_range": None,
            "cl_level": 0.95,
        },
        "systematics": {
            "enable": False,
            "sigma_shifts": {
                "slope": 0.05,
                "intercept": 0.05,
                "eff_po214": 0.1,
                "eff_po218": 0.1,
            },
        },
        "pipeline": {},
        "spectral_fit": {},
        "time_fit": {},
        "plotting": {},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    loaded = load_config(str(p))
    assert loaded["adc"]["min_channel"] == 0
    assert loaded["efficiency"]["eff_po214"] == 0.4


def test_load_events(tmp_path):
    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3],
            "fBits": [0, 0, 0],
            "timestamp": [1000, 1005, 1010],
            "adc": [1200, 1300, 1250],
            "fchannel": [1, 1, 1],
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    loaded = load_events(str(p))
    assert np.array_equal(loaded["timestamp"].values, np.array([1000, 1005, 1010]))
    assert np.array_equal(loaded["adc"].values, np.array([1200, 1300, 1250]))


def test_write_summary_and_copy_config(tmp_path):
    summary = {"a": 1, "b": 2}
    outdir = tmp_path / "out"
    ts = "19700101T000000Z"
    results = write_summary(str(outdir), summary, ts)
    assert (Path(results) / "summary.json").exists()
    # Create dummy config and copy
    cfg = {"test": 1}
    cp = tmp_path / "cfg.json"
    with open(cp, "w") as f:
        json.dump(cfg, f)
    dest = copy_config(str(outdir), str(cp))
    assert Path(dest).exists()


def test_write_summary_with_nullable_integers(tmp_path):
    series = pd.Series([1, pd.NA], dtype="Int64")
    summary = {"present": series.iloc[0], "missing": series.iloc[1], "list": series.tolist()}
    outdir = tmp_path / "out2"
    ts = "19700101T000001Z"
    results = write_summary(str(outdir), summary, ts)
    summary_path = Path(results) / "summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["present"] == 1
    assert loaded["missing"] is None
    assert loaded["list"] == [1, None]
