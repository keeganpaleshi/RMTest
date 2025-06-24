import numpy as np
import pandas as pd
from datetime import datetime, timezone
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline
from io_utils import load_events


def test_rate_histogram_datetime_column():
    ts = pd.date_range("1970-01-01", periods=5, freq="s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "adc": np.arange(5)})
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
    bins = np.arange(0, 7)
    rate, live = baseline.rate_histogram(df, bins)
    assert live == pytest.approx(4.0)
    assert np.allclose(rate, np.histogram(df["adc"], bins=bins)[0] / live)


def test_subtract_baseline_datetime_column():
    ts_an = pd.date_range("1970-01-01", periods=5, freq="s", tz="UTC")
    df_an = pd.DataFrame({"timestamp": ts_an, "adc": [1, 2, 3, 4, 5]})
    assert df_an["timestamp"].dtype == "datetime64[ns, UTC]"
    ts_bl = pd.to_datetime(np.linspace(86400, 86440, 50), unit="s", utc=True)
    df_bl = pd.DataFrame({"timestamp": ts_bl, "adc": np.tile([1,2,3,4,5],10)})
    df_full = pd.concat([df_an, df_bl], ignore_index=True)
    bins = np.arange(0, 7)
    out = baseline.subtract_baseline(
        df_an,
        df_full,
        bins=bins,
        t_base0=datetime(1970,1,2,0,0,0,tzinfo=timezone.utc),
        t_base1=datetime(1970,1,2,0,0,40,tzinfo=timezone.utc),
        mode="all",
    )
    integral = out["subtracted_adc_hist"].iloc[0].sum()
    assert integral == pytest.approx(0.0, rel=1e-6)
    assert out["timestamp"].dtype == "datetime64[ns, UTC]"


def test_subtract_baseline_loaded_from_csv(tmp_path):
    df_an = pd.DataFrame(
        {
            "fUniqueID": [1, 2, 3],
            "fBits": [0, 0, 0],
            "timestamp": [0.0, 1.0, 2.0],
            "adc": [1, 2, 3],
            "fchannel": [1, 1, 1],
        }
    )
    df_full = pd.DataFrame(
        {
            "fUniqueID": [4, 5, 6, 7],
            "fBits": [0, 0, 0, 0],
            "timestamp": [100.0, 110.0, 120.0, 130.0],
            "adc": [1, 2, 3, 4],
            "fchannel": [1, 1, 1, 1],
        }
    )
    p_an = tmp_path / "an.csv"
    p_full = tmp_path / "full.csv"
    df_an.to_csv(p_an, index=False)
    df_full.to_csv(p_full, index=False)

    an_loaded = load_events(p_an)
    full_loaded = load_events(p_full)

    out = baseline.subtract_baseline(
        an_loaded,
        pd.concat([an_loaded, full_loaded], ignore_index=True),
        bins=np.arange(0, 7),
        t_base0=100.0,
        t_base1=130.0,
    )

    assert out["timestamp"].dtype == "datetime64[ns, UTC]"


def test_subtract_baseline_string_times(tmp_path):
    df = pd.DataFrame(
        {
            "fUniqueID": [1, 2],
            "fBits": [0, 0],
            "timestamp": ["1970-01-01T00:00:00Z", "1970-01-01T00:00:01Z"],
            "adc": [1, 2],
            "fchannel": [1, 1],
        }
    )
    p = tmp_path / "str.csv"
    df.to_csv(p, index=False)
    loaded = load_events(p)

    out = baseline.subtract_baseline(
        loaded,
        loaded,
        bins=np.arange(0, 3),
        t_base0="1970-01-01T00:00:00Z",
        t_base1="1970-01-01T00:00:01Z",
        mode="none",
    )

    assert out["timestamp"].dtype == "datetime64[ns, UTC]"

