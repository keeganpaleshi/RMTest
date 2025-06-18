import os
import json
import tempfile
from pathlib import Path
import sys
import logging
import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from io_utils import (
    load_config,
    load_events,
    write_summary,
    copy_config,
    apply_burst_filter,
)


def test_load_config(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "spectral_fit": {
            "expected_peaks": {"Po210": 1250, "Po218": 1400, "Po214": 1800}
        },
        "time_fit": {"do_time_fit": True},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
        "burst_filter": {"burst_mode": "rate"},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    loaded = load_config(p)
    assert loaded["pipeline"]["log_level"] == "INFO"
    assert loaded["burst_filter"]["burst_mode"] == "rate"


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
    cfg_loaded = load_config(p)
    assert "spectral_fit" in cfg_loaded


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
    assert loaded["timestamp"].dtype == float
    assert np.array_equal(loaded["timestamp"].values, np.array([1000.0, 1005.0, 1010.0]))
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
    assert loaded["timestamp"].dtype == float
    assert np.array_equal(loaded["timestamp"].values, np.array([1000.0, 1005.0, 1020.0]))
    assert "3 discarded" in caplog.text


def test_load_events_column_aliases(tmp_path):
    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "time": [1000],
            "adc_ch": [1250],
            "fchannel": [1],
        }
    )
    p = tmp_path / "alias.csv"
    df.to_csv(p, index=False)
    loaded = load_events(p)
    assert loaded["timestamp"].dtype == float
    assert list(loaded["timestamp"])[0] == 1000.0
    assert list(loaded["adc"])[0] == 1250
    assert "time" not in loaded.columns
    assert "adc_ch" not in loaded.columns


def test_load_events_missing_column(tmp_path):
    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [1000],
            # ADC column intentionally missing
            "fchannel": [1],
        }
    )
    p = tmp_path / "missing.csv"
    df.to_csv(p, index=False)
    with pytest.raises(KeyError):
        load_events(p)


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
    dest = copy_config(results, cp)
    assert Path(dest).exists()


def test_write_summary_with_nullable_integers(tmp_path):
    series = pd.Series([1, pd.NA], dtype="Int64")
    summary = {"present": series.iloc[0], "missing": series.iloc[1], "list": series.tolist()}
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
    cfg = {"burst_filter": {"burst_window_size_s": 10, "rolling_median_window": 3, "burst_multiplier": 3}}
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
    cfg = {"burst_filter": {"burst_window_size_s": 10, "rolling_median_window": 3, "burst_multiplier": 3}}
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


def test_apply_burst_filter_histogram_called(monkeypatch):
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
    orig_hist = np.histogram

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig_hist(*args, **kwargs)

    monkeypatch.setattr(np, "histogram", wrapped)
    apply_burst_filter(df, cfg, mode="micro")
    assert calls["n"] == 1


def test_apply_burst_filter_both_matches_sequential():
    base = np.concatenate([np.arange(120), np.arange(130, 150)])
    micro = np.full(6, 30)
    rate = np.concatenate([
        120 + i + 0.1 * np.arange(4)
        for i in range(10)
    ]).ravel()
    times = np.concatenate([base, micro, rate])
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
            "micro_window_size_s": 1,
            "micro_count_threshold": 5,
        }
    }

    seq, rem1 = apply_burst_filter(df, cfg, mode="micro")
    seq, rem2 = apply_burst_filter(seq, cfg, mode="rate")
    both, total = apply_burst_filter(df, cfg, mode="both")
    assert total == rem1 + rem2
    assert both.reset_index(drop=True).equals(seq.reset_index(drop=True))


def test_load_config_duplicate_keys(tmp_path):
    txt = (
        "{"
        "\n  \"pipeline\": {\"log_level\": \"INFO\"},"
        "\n  \"pipeline\": {\"log_level\": \"DEBUG\"},"
        "\n  \"spectral_fit\": {\"expected_peaks\": {\"Po210\": 1}},"
        "\n  \"time_fit\": {\"do_time_fit\": true},"
        "\n  \"systematics\": {\"enable\": false},"
        "\n  \"plotting\": {\"plot_save_formats\": [\"png\"]}"
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


def test_load_config_unknown_key(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "spectral_fit": {},
        "time_fit": {"do_time_fit": True},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
        "burts_filter": {},
    }
    p = tmp_path / "cfg.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        load_config(p)

