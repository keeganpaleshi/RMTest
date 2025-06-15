import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from io_utils import (
    apply_burst_filter,
    copy_config,
    load_config,
    load_events,
    write_summary,
)


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
        "systematics": {"enable": False},
        "pipeline": {"log_level": "INFO"},
        "spectral_fit": {
            "expected_peaks": {"Po210": 1250, "Po218": 1400, "Po214": 1800}
        },
        "time_fit": {"do_time_fit": True},
        "plotting": {"plot_save_formats": ["png"]},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    loaded = load_config(p)
    assert loaded["adc"]["min_channel"] == 0
    assert loaded["efficiency"]["eff_po214"] == 0.4


def test_load_config_missing_key(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "spectral_fit": {},
        "time_fit": {"do_time_fit": True},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    with pytest.raises(KeyError):
        load_config(p)


def test_load_config_missing_section(tmp_path):
    cfg = {
        "spectral_fit": {"expected_peaks": {"Po210": 1}},
        "time_fit": {"do_time_fit": True},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    with pytest.raises(KeyError):
        load_config(p)


def test_load_events(tmp_path, caplog):
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
    with caplog.at_level(logging.INFO):
        loaded = load_events(p)
    assert np.array_equal(loaded["timestamp"].values, np.array([1000, 1005, 1010]))
    assert np.array_equal(loaded["adc"].values, np.array([1200, 1300, 1250]))
    assert "0 discarded" in caplog.text


def test_load_events_drop_bad_rows(tmp_path, caplog):
    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 2, 3, 4, 5],
            "fBits": [0, 0, 0, 0, 0, 0],
            "timestamp": [1000, 1005, 1005, 1010, np.nan, 1020],
            "adc": [1200, 1300, 1300, np.inf, 1350, 1250],
            "fchannel": [1, 1, 1, 1, 1, 1],
        }
    )
    p = tmp_path / "data_bad.csv"
    df.to_csv(p, index=False)
    with caplog.at_level(logging.INFO):
        loaded = load_events(p)
    # Expect rows with NaN/inf removed and duplicate dropped
    assert np.array_equal(loaded["timestamp"].values, np.array([1000, 1005, 1020]))
    assert "3 discarded" in caplog.text


def test_write_summary_and_copy_config(tmp_path):
    summary = {"a": 1, "b": 2}
    outdir = tmp_path / "out"
    ts = "19700101T000000Z"
    results = write_summary(outdir, summary, ts)
    assert (Path(results) / "summary.json").exists()
    # Create dummy config and copy
    cfg = {"test": 1}
    cp = tmp_path / "cfg.json"
    with open(cp, "w") as f:
        json.dump(cfg, f)
    dest = copy_config(outdir, cp)
    assert Path(dest).exists()


def test_write_summary_with_nullable_integers(tmp_path):
    series = pd.Series([1, pd.NA], dtype="Int64")
    summary = {
        "present": series.iloc[0],
        "missing": series.iloc[1],
        "list": series.tolist(),
    }
    outdir = tmp_path / "out2"
    ts = "19700101T000001Z"
    results = write_summary(outdir, summary, ts)
    summary_path = Path(results) / "summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["present"] == 1
    assert loaded["missing"] is None
    assert loaded["list"] == [1, None]


def test_write_summary_with_nan_values(tmp_path):
    summary = {"nan": float("nan"), "inf": float("inf"), "list": [float("nan"), 1.0]}
    outdir = tmp_path / "out_nan"
    ts = "19700101T000002Z"
    results = write_summary(outdir, summary, ts)
    summary_path = Path(results) / "summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["nan"] is None
    assert loaded["inf"] is None
    assert loaded["list"] == [None, 1.0]


def test_apply_burst_filter_no_removal():
    df = pd.DataFrame(
        {
            "fUniqueID": range(100),
            "fBits": [0] * 100,
            "timestamp": np.arange(100),
            "adc": [1000] * 100,
            "fchannel": [1] * 100,
        }
    )
    cfg = {
        "burst_filter": {
            "burst_window_size_s": 10,
            "rolling_median_window": 3,
            "burst_multiplier": 3,
        }
    }
    filtered, removed = apply_burst_filter(df, cfg)
    assert len(filtered) == 100
    assert removed == 0


def test_apply_burst_filter_with_burst():
    base_times = np.arange(100)
    burst_times = np.full(50, 30)
    times = np.concatenate([base_times, burst_times])
    df = pd.DataFrame(
        {
            "fUniqueID": range(len(times)),
            "fBits": [0] * len(times),
            "timestamp": times,
            "adc": [1000] * len(times),
            "fchannel": [1] * len(times),
        }
    )
    cfg = {
        "burst_filter": {
            "burst_window_size_s": 10,
            "rolling_median_window": 3,
            "burst_multiplier": 3,
        }
    }
    filtered, removed = apply_burst_filter(df, cfg)
    assert removed == 60
    assert len(filtered) == len(times) - 60


def test_apply_burst_filter_mode_none():
    df = pd.DataFrame(
        {
            "fUniqueID": range(10),
            "fBits": [0] * 10,
            "timestamp": [0] * 10,
            "adc": [1000] * 10,
            "fchannel": [1] * 10,
        }
    )
    cfg = {
        "burst_filter": {
            "burst_window_size_s": 1,
            "rolling_median_window": 1,
            "burst_multiplier": 2,
            "micro_window_size_s": 0.1,
            "micro_count_threshold": 2,
        }
    }
    filtered, removed = apply_burst_filter(df, cfg, mode="none")
    assert len(filtered) == len(df)
    assert removed == 0


def test_apply_burst_filter_micro_burst():
    times = np.concatenate([np.arange(10), np.full(4, 20)])
    df = pd.DataFrame(
        {
            "fUniqueID": range(len(times)),
            "fBits": [0] * len(times),
            "timestamp": times,
            "adc": [1000] * len(times),
            "fchannel": [1] * len(times),
        }
    )
    cfg = {
        "burst_filter": {
            "micro_window_size_s": 1,
            "micro_count_threshold": 3,
        }
    }
    filtered, removed = apply_burst_filter(df, cfg, mode="micro")
    assert removed == 4
    assert len(filtered) == len(times) - 4


def test_apply_burst_filter_single_searchsorted(monkeypatch):
    times = np.arange(100)
    df = pd.DataFrame(
        {
            "fUniqueID": range(len(times)),
            "fBits": [0] * len(times),
            "timestamp": times,
            "adc": [1000] * len(times),
            "fchannel": [1] * len(times),
        }
    )
    cfg = {
        "burst_filter": {
            "micro_window_size_s": 1,
            "micro_count_threshold": 2,
        }
    }

    calls = {"n": 0}
    orig_ss = np.searchsorted

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig_ss(*args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", wrapped)
    apply_burst_filter(df, cfg, mode="micro")
    assert calls["n"] == 1


def test_load_config_duplicate_keys(tmp_path):
    txt = (
        "{"
        '\n  "pipeline": {"log_level": "INFO"},'
        '\n  "pipeline": {"log_level": "DEBUG"},'
        '\n  "spectral_fit": {"expected_peaks": {"Po210": 1}},'
        '\n  "time_fit": {"do_time_fit": true},'
        '\n  "systematics": {"enable": false},'
        '\n  "plotting": {"plot_save_formats": ["png"]}'
        "\n}"
    )
    p = tmp_path / "dup.json"
    with open(p, "w") as f:
        f.write(txt)
    with pytest.raises(ValueError, match="Duplicate key"):
        load_config(p)


def test_load_config_invalid_half_life(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "spectral_fit": {"expected_peaks": {"Po210": 1}},
        "time_fit": {"do_time_fit": True, "hl_po214": -1.0},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        load_config(p)


def test_load_config_invalid_baseline(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "baseline": {"monitor_volume_l": -1, "sample_volume_l": 0.0},
        "spectral_fit": {"expected_peaks": {"Po210": 1}},
        "time_fit": {"do_time_fit": True},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        load_config(p)


def test_load_config_invalid_burst_filter(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "burst_filter": {"burst_window_size_s": -5},
        "spectral_fit": {"expected_peaks": {"Po210": 1}},
        "time_fit": {"do_time_fit": True},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        load_config(p)
