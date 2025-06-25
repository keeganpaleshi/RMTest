import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io_utils
import baseline
import analyze


def test_datetime_pipeline(tmp_path):
    ts_iso = [
        "1970-01-01T00:00:00Z",
        "1970-01-01T00:00:01Z",
        "1970-01-01T00:00:02Z",
    ]
    df = pd.DataFrame({
        "fUniqueID": [1, 2, 3],
        "fBits": [0, 0, 0],
        "timestamp": ts_iso,
        "adc": [0, 1, 2],
        "fchannel": [1, 1, 1],
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    loaded = io_utils.load_events(csv_path)
    assert str(loaded["timestamp"].dtype) == "datetime64[ns, UTC]"

    bins = np.arange(0, 5)
    subtracted, _ = baseline.subtract(
        loaded,
        loaded,
        bins=bins,
        t_base0=datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        t_base1=datetime(1970, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        mode="all",
    )
    assert str(subtracted["timestamp"].dtype) == "datetime64[ns, UTC]"

    args = argparse.Namespace(slope=None)
    analysis_df, *_ = analyze.prepare_analysis_df(
        subtracted,
        spike_end=None,
        spike_periods=[],
        run_periods=[],
        analysis_end=None,
        t0_global=datetime(1970, 1, 1, tzinfo=timezone.utc),
        cfg={},
        args=args,
    )
    assert str(analysis_df["timestamp"].dtype) == "datetime64[ns, UTC]"
