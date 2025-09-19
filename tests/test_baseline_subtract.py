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
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(0, 9, 10)],
        "adc": np.arange(10),
    })
    bins = np.arange(0, 11)
    out_df, hist = baseline_utils.subtract_baseline_dataframe(
        df,
        df,
        bins=bins,
        t_base0=pd.Timestamp(0, unit="s", tz="UTC"),
        t_base1=pd.Timestamp(5, unit="s", tz="UTC"),
        mode="none",
    )
    counts_before, _ = np.histogram(df["adc"], bins=bins)
    assert np.array_equal(counts_before, hist)


def test_baseline_time_norm():
    df_an = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(0, 4, 5)],
        "adc": [1, 2, 3, 4, 5],
    })
    df_bl = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(100, 140, 50)],
        "adc": np.tile([1, 2, 3, 4, 5], 10),
    })
    df_full = pd.concat([df_an, df_bl], ignore_index=True)
    bins = np.arange(0, 7)
    out_df, hist = baseline_utils.subtract_baseline_dataframe(
        df_an,
        df_full,
        bins=bins,
        t_base0=pd.Timestamp(100, unit="s", tz="UTC"),
        t_base1=pd.Timestamp(140, unit="s", tz="UTC"),
        mode="all",
    )
    integral = hist.sum()
    assert integral == pytest.approx(0.0, rel=1e-6)


def test_baseline_mode_electronics():
    df_an = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(0, 4, 5)],
        "adc": [1, 2, 3, 4, 5],
    })
    df_bl = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(100, 140, 50)],
        "adc": np.tile([1, 2, 3, 4, 5], 10),
    })
    df_full = pd.concat([df_an, df_bl], ignore_index=True)
    bins = np.arange(0, 7)
    out_df, hist = baseline_utils.subtract_baseline_dataframe(
        df_an,
        df_full,
        bins=bins,
        t_base0=pd.Timestamp(100, unit="s", tz="UTC"),
        t_base1=pd.Timestamp(140, unit="s", tz="UTC"),
        mode="electronics",
    )
    assert out_df is not df_an
    assert hist.sum() == pytest.approx(0.0, rel=1e-6)


def test_baseline_mode_radon_retains_electronics():
    df_an = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(0, 4, 5)],
        "adc": [1, 2, 3, 4, 5],
    })
    df_bl = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(100, 140, 50)],
        "adc": np.tile([1, 2, 3, 4, 5], 10),
    })
    df_full = pd.concat([df_an, df_bl], ignore_index=True)
    bins = np.arange(0, 7)
    out_df, hist = baseline_utils.subtract_baseline_dataframe(
        df_an,
        df_full,
        bins=bins,
        t_base0=pd.Timestamp(100, unit="s", tz="UTC"),
        t_base1=pd.Timestamp(140, unit="s", tz="UTC"),
        mode="radon",
    )
    counts_before, _ = np.histogram(df_an["adc"], bins=bins)
    assert np.array_equal(counts_before, hist)


def test_baseline_none_datetime():
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(0, 9, 10)],
        "adc": np.arange(10),
    })
    bins = np.arange(0, 11)
    out_df, hist = baseline_utils.subtract_baseline_dataframe(
        df,
        df,
        bins=bins,
        t_base0=datetime.fromtimestamp(0, tz=timezone.utc),
        t_base1=datetime.fromtimestamp(5, tz=timezone.utc),
        mode="none",
    )
    counts_before, _ = np.histogram(df["adc"], bins=bins)
    assert np.array_equal(counts_before, hist)


def test_baseline_time_norm_datetime():
    df_an = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(0, 4, 5)],
        "adc": [1, 2, 3, 4, 5],
    })
    df_bl = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in np.linspace(100, 140, 50)],
        "adc": np.tile([1, 2, 3, 4, 5], 10),
    })
    df_full = pd.concat([df_an, df_bl], ignore_index=True)
    bins = np.arange(0, 7)
    out_df, hist = baseline_utils.subtract_baseline_dataframe(
        df_an,
        df_full,
        bins=bins,
        t_base0=datetime.fromtimestamp(100, tz=timezone.utc),
        t_base1=datetime.fromtimestamp(140, tz=timezone.utc),
        mode="all",
    )
    integral = hist.sum()
    assert integral == pytest.approx(0.0, rel=1e-6)

