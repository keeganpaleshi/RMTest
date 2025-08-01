import numpy as np
import pandas as pd
from datetime import datetime, timezone
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline


def test_rate_histogram_datetime_column():
    ts = pd.date_range("1970-01-01", periods=5, freq="s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "adc": np.arange(5)})
    bins = np.arange(0, 7)
    rate, live = baseline.rate_histogram(df, bins)
    assert live == pytest.approx(4.0)
    assert np.allclose(rate, np.histogram(df["adc"], bins=bins)[0] / live)
    assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_subtract_baseline_datetime_column():
    ts_an = pd.date_range("1970-01-01", periods=5, freq="s", tz="UTC")
    df_an = pd.DataFrame({"timestamp": ts_an, "adc": [1, 2, 3, 4, 5]})
    ts_bl = [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(86400, 86440, 50)]
    df_bl = pd.DataFrame({"timestamp": ts_bl, "adc": np.tile([1,2,3,4,5],10)})
    df_full = pd.concat([df_an, df_bl], ignore_index=True)
    bins = np.arange(0, 7)
    out_df, (hist, _) = baseline.subtract(
        df_an,
        df_full,
        bins=bins,
        t_base0=datetime(1970,1,2,0,0,0,tzinfo=timezone.utc),
        t_base1=datetime(1970,1,2,0,0,40,tzinfo=timezone.utc),
        mode="all",
    )
    integral = hist.sum()
    assert integral == pytest.approx(0.0, rel=1e-6)
    assert str(out_df["timestamp"].dtype) == "datetime64[ns, UTC]"

