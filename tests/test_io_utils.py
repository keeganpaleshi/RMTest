import os
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from io_utils import load_config, load_data, write_summary, copy_config


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
            "fit_po214_only": true,
            "fit_po218_po214": false,
            "fix_B": false,
            "fix_N0": false,
            "baseline_range": null,
            "cl_level": 0.95,
        },
        "systematics": {
            "enable": false,
            "sigma_shifts": {
                "slope": 0.05,
                "intercept": 0.05,
                "eff_po214": 0.1,
                "eff_po218": 0.1,
            },
        },
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    loaded = load_config(str(p))
    assert loaded["adc"]["min_channel"] == 0
    assert loaded["efficiency"]["eff_po214"] == 0.4


def test_load_data(tmp_path):
    # Create small CSV
    df = pd.DataFrame(
        {"timestamp": [1000, 1005, 1010], "adc": [1200, 1300, 1250]})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    ts, adc = load_data(str(p))
    assert np.array_equal(ts, np.array([1000, 1005, 1010]))
    assert np.array_equal(adc, np.array([1200, 1300, 1250]))


def test_write_summary_and_copy_config(tmp_path):
    summary = {"a": 1, "b": 2}
    outdir = tmp_path / "out"
    write_summary(str(outdir), summary)
    assert (outdir / "summary.json").exists()
    # Create dummy config and copy
    cfg = {"test": 1}
    cp = tmp_path / "cfg.json"
    with open(cp, "w") as f:
        json.dump(cfg, f)
    copy_config(str(outdir), str(cp))
    assert (outdir / "cfg.json").exists()
