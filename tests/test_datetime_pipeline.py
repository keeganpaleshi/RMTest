import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline
import analyze
from io_utils import load_events


def test_iso_datetime_pipeline(tmp_path):
    # Create CSV with ISO-formatted timestamps
    iso_times = [
        "1970-01-01T00:00:00Z",
        "1970-01-01T00:00:10Z",
        "1970-01-01T00:00:20Z",
    ]
    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3],
            "fBits": [0, 0, 0],
            "timestamp": iso_times,
            "adc": [1000, 1001, 1002],
            "fchannel": [1, 1, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    # Load events and ensure timezone-aware dtype
    loaded = load_events(csv_path)
    assert str(loaded["timestamp"].dtype) == "datetime64[ns, UTC]"

    # Baseline subtraction
    bins = np.arange(0, 10)
    df_bl = baseline.subtract_baseline(
        loaded,
        loaded,
        bins=bins,
        t_base0=datetime(1970, 1, 1, tzinfo=timezone.utc),
        t_base1=datetime(1970, 1, 1, 0, 0, 20, tzinfo=timezone.utc),
        mode="all",
    )
    assert str(df_bl["timestamp"].dtype) == "datetime64[ns, UTC]"

    # Prepare analysis DataFrame
    args = argparse.Namespace(slope=None)
    out_df, *_ = analyze.prepare_analysis_df(
        df_bl,
        spike_end=None,
        spike_periods=[],
        run_periods=[],
        analysis_end=None,
        t0_global=datetime(1970, 1, 1, tzinfo=timezone.utc),
        cfg={},
        args=args,
    )
    assert str(out_df["timestamp"].dtype) == "datetime64[ns, UTC]"
