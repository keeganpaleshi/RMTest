import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline
import analyze
from io_utils import load_events


def test_load_events_returns_timezone(tmp_path):
    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [1000],
        "adc": [1200],
        "fchannel": [1],
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    loaded = load_events(p)
    assert str(loaded["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_subtract_baseline_preserves_dtype():
    ts = pd.date_range("1970-01-01", periods=3, freq="s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "adc": [1, 2, 3]})
    bins = np.arange(0, 5)
    out = baseline.subtract_baseline(
        df,
        df,
        bins=bins,
        t_base0=datetime(1970, 1, 1, tzinfo=timezone.utc),
        t_base1=datetime(1970, 1, 1, 0, 2, tzinfo=timezone.utc),
        mode="none",
    )
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_prepare_analysis_df_preserves_dtype():
    ts = pd.date_range("1970-01-01", periods=5, freq="s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "adc": np.arange(5)})
    args = argparse.Namespace(slope=None)
    out_df, *_ = analyze.prepare_analysis_df(
        df,
        spike_end=None,
        spike_periods=[],
        run_periods=[],
        analysis_end=None,
        t0_global=datetime(1970, 1, 1, tzinfo=timezone.utc),
        cfg={},
        args=args,
    )
    assert str(out_df["timestamp"].dtype) == "datetime64[ns, UTC]"
