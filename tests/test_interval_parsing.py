import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
from utils import parse_time_arg
from utils.time_utils import parse_timestamp
from dateutil.tz import gettz
import pandas as pd


def test_cli_interval_parsing_to_datetime():
    args = analyze.parse_args([
        "--config", "cfg.json",
        "--input", "data.csv",
        "--output_dir", "out",
        "--baseline_range", "1970-01-01T00:00:01Z", "1970-01-01T00:00:02Z",
        "--radon-interval", "1970-01-01T00:00:03Z", "1970-01-01T00:00:04Z",
    ])
    tzinfo = gettz(args.timezone)
    args.baseline_range = [parse_time_arg(t, tz=tzinfo) for t in args.baseline_range]
    args.radon_interval = [parse_time_arg(t, tz=tzinfo) for t in args.radon_interval]
    for dt in args.baseline_range + args.radon_interval:
        assert dt.tzinfo is not None
        assert dt.tzinfo.utcoffset(dt) == timedelta(0)


def test_config_interval_parsing_to_datetime():
    cfg = {
        "baseline": {"range": ["1970-01-01T00:00:01Z", "1970-01-01T00:00:02Z"]},
        "analysis": {"radon_interval": ["1970-01-01T00:00:03Z", "1970-01-01T00:00:04Z"]},
    }
    b_start = parse_timestamp(cfg["baseline"]["range"][0])
    b_end = parse_timestamp(cfg["baseline"]["range"][1])
    r_start = parse_timestamp(cfg["analysis"]["radon_interval"][0])
    r_end = parse_timestamp(cfg["analysis"]["radon_interval"][1])
    for dt in (b_start, b_end, r_start, r_end):
        assert dt.tzinfo is not None
        assert dt.tzinfo.utcoffset(dt) == timedelta(0)

def test_parse_allow_negative_activity():
    args = analyze.parse_args([
        "--config", "cfg.json",
        "--input", "data.csv",
        "--output_dir", "out",
        "--allow-negative-activity",
    ])
    assert args.allow_negative_activity is True
