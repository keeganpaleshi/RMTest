import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline
import analyze
from io_utils import load_events


def test_pipeline_preserves_timestamp_dtype(tmp_path):
    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": [
            "1970-01-01T00:00:00Z",
            "1970-01-01T00:00:01Z",
            "1970-01-01T00:00:02Z",
        ],
        "adc": [10.0, 11.0, 12.0],
        "fchannel": [1, 1, 1],
    })
    csv_path = tmp_path / "events.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_events(csv_path)
    assert str(loaded["timestamp"].dtype) == "datetime64[ns, UTC]"

    bins = np.arange(0, 20, 10)
    sub = baseline.subtract_baseline(
        loaded,
        loaded,
        bins=bins,
        t_base0=pd.Timestamp("1970-01-01T00:00:00Z"),
        t_base1=pd.Timestamp("1970-01-01T00:00:02Z"),
        mode="none",
    )
    assert str(sub["timestamp"].dtype) == "datetime64[ns, UTC]"

    args = argparse.Namespace(slope=None)
    out_df, *_ = analyze.prepare_analysis_df(
        sub,
        spike_end=None,
        spike_periods=[],
        run_periods=[],
        analysis_end=None,
        t0_global=pd.Timestamp("1970-01-01T00:00:00Z"),
        cfg={},
        args=args,
    )
    assert str(out_df["timestamp"].dtype) == "datetime64[ns, UTC]"
