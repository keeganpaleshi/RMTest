import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline
import baseline_utils


def test_baseline_none():
    df = pd.DataFrame({
        "timestamp": np.linspace(0, 9, 10),
        "adc": np.arange(10),
    })
    bins = np.arange(0, 11)
    out = baseline_utils.subtract_baseline_dataframe(
        df,
        df,
        bins=bins,
        t_base0=pd.Timestamp(0, unit="s", tz="UTC"),
        t_base1=pd.Timestamp(5, unit="s", tz="UTC"),
        mode="none",
    )
    hist_before, _ = baseline.rate_histogram(df, bins)
    hist_after, _ = baseline.rate_histogram(out, bins)
    assert np.allclose(hist_before, hist_after)
    assert out["timestamp"].dtype == "datetime64[ns, UTC]"


def test_baseline_time_norm():
    df_an = pd.DataFrame({
        "timestamp": np.linspace(0, 4, 5),
        "adc": [1, 2, 3, 4, 5],
    })
    df_bl = pd.DataFrame({
        "timestamp": np.linspace(100, 140, 50),
        "adc": np.tile([1, 2, 3, 4, 5], 10),
    })
    df_full = pd.concat([df_an, df_bl], ignore_index=True)
    bins = np.arange(0, 7)
    out = baseline_utils.subtract_baseline_dataframe(
        df_an,
        df_full,
        bins=bins,
        t_base0=pd.Timestamp(100, unit="s", tz="UTC"),
        t_base1=pd.Timestamp(140, unit="s", tz="UTC"),
        mode="all",
    )
    integral = out["subtracted_adc_hist"].iloc[0].sum()
    assert integral == pytest.approx(0.0, rel=1e-6)
    assert out["timestamp"].dtype == "datetime64[ns, UTC]"


def test_baseline_none_datetime():
    df = pd.DataFrame({
        "timestamp": np.linspace(0, 9, 10),
        "adc": np.arange(10),
    })
    bins = np.arange(0, 11)
    out = baseline_utils.subtract_baseline_dataframe(
        df,
        df,
        bins=bins,
        t_base0=datetime.fromtimestamp(0, tz=timezone.utc),
        t_base1=datetime.fromtimestamp(5, tz=timezone.utc),
        mode="none",
    )
    hist_before, _ = baseline.rate_histogram(df, bins)
    hist_after, _ = baseline.rate_histogram(out, bins)
    assert np.allclose(hist_before, hist_after)
    assert out["timestamp"].dtype == "datetime64[ns, UTC]"


def test_baseline_time_norm_datetime():
    df_an = pd.DataFrame({
        "timestamp": np.linspace(0, 4, 5),
        "adc": [1, 2, 3, 4, 5],
    })
    df_bl = pd.DataFrame({
        "timestamp": np.linspace(100, 140, 50),
        "adc": np.tile([1, 2, 3, 4, 5], 10),
    })
    df_full = pd.concat([df_an, df_bl], ignore_index=True)
    bins = np.arange(0, 7)
    out = baseline_utils.subtract_baseline_dataframe(
        df_an,
        df_full,
        bins=bins,
        t_base0=datetime.fromtimestamp(100, tz=timezone.utc),
        t_base1=datetime.fromtimestamp(140, tz=timezone.utc),
        mode="all",
    )
    integral = out["subtracted_adc_hist"].iloc[0].sum()
    assert integral == pytest.approx(0.0, rel=1e-6)
    assert out["timestamp"].dtype == "datetime64[ns, UTC]"


def test_baseline_numeric_timestamps():
    df = pd.DataFrame({"timestamp": [0.0, 1.0, 2.0], "adc": [1, 2, 3]})
    bins = np.arange(0, 4)
    out = baseline_utils.subtract_baseline_dataframe(
        df, df, bins, t_base0=0.0, t_base1=2.0
    )
    assert out["timestamp"].dtype == "datetime64[ns, UTC]"


def test_baseline_string_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": [
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:01Z",
                "1970-01-01T00:00:02Z",
            ],
            "adc": [1, 2, 3],
        }
    )
    bins = np.arange(0, 4)
    out = baseline_utils.subtract_baseline_dataframe(
        df,
        df,
        bins,
        t_base0="1970-01-01T00:00:00Z",
        t_base1="1970-01-01T00:00:02Z",
        mode="none",
    )
    assert out["timestamp"].dtype == "datetime64[ns, UTC]"

