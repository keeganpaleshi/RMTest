import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline
from io_utils import load_events


def test_load_events_mixed_timestamp_types(tmp_path):
    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3, 4],
            "fBits": [0, 0, 0, 0],
            "timestamp": ["1000.0", 1005.5, "1970-01-01T00:16:47Z", "1007"],
            "adc": [1200, 1300, 1250, 1220],
            "fchannel": [1, 1, 1, 1],
        }
    )
    p = tmp_path / "mix.csv"
    df.to_csv(p, index=False)
    loaded = load_events(p)
    assert str(loaded["timestamp"].dtype) == "datetime64[ns, UTC]"
    expected = pd.Series([
        pd.to_datetime(1000.0, unit="s", utc=True),
        pd.to_datetime(1005.5, unit="s", utc=True),
        pd.to_datetime("1970-01-01T00:16:47Z", utc=True),
        pd.to_datetime(1007.0, unit="s", utc=True),
    ])
    assert np.array_equal(
        loaded["timestamp"].view("int64"), expected.view("int64")
    )


def test_subtract_baseline_df_float_args():
    ts = pd.date_range("1970-01-02", periods=5, freq="s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "adc": np.arange(5)})
    bins = np.arange(0, 7)
    t0 = float(ts[0].timestamp())
    t1 = float(ts[-1].timestamp())
    out = baseline.subtract_baseline_df(
        df,
        df,
        bins=bins,
        t_base0=t0,
        t_base1=t1,
        mode="none",
    )
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_subtract_baseline_df_string_args():
    ts = pd.date_range("1970-01-03", periods=5, freq="s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "adc": np.arange(5)})
    bins = np.arange(0, 7)
    t0 = ts[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    t1 = ts[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
    out = baseline.subtract_baseline_df(
        df,
        df,
        bins=bins,
        t_base0=t0,
        t_base1=t1,
        mode="none",
    )
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
