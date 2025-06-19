import numpy as np
import logging

__all__ = ["rate_histogram", "subtract_baseline"]


def rate_histogram(df, bins):
    """Return (histogram in counts/s, live_time_s)"""
    if df.empty:
        return np.zeros(len(bins) - 1, dtype=float), 0.0
    t_end = df["timestamp"].iloc[-1]
    t_start = df["timestamp"].iloc[0]
    if isinstance(t_end, np.datetime64):
        live = float((t_end - t_start) / np.timedelta64(1, "s"))
    else:
        live = float(t_end - t_start)
    hist_src = df.get("subtracted_adc_hist", df["adc"]).to_numpy()
    hist, _ = np.histogram(hist_src, bins=bins)
    if live <= 0:
        return np.zeros_like(hist, dtype=float), live
    return hist / live, live


def subtract_baseline(df_analysis, df_full, bins, t_base0, t_base1,
                      mode="all", live_time_analysis=None):
    """Subtract baseline from df_analysis and return NEW dataframe copy.

    mode: 'none' | 'electronics' | 'radon' | 'all'
    live_time_analysis: seconds represented by df_analysis;
                        if None it is calculated internally.
    """
    assert mode in ("none", "electronics", "radon", "all")

    if mode == "none":
        return df_analysis.copy()

    # analysis spectrum (counts/s)
    rate_an, live_an = rate_histogram(df_analysis, bins)
    if live_time_analysis is None:
        live_time_analysis = live_an

    # baseline slice
    mask = (df_full["timestamp"] >= t_base0) & (df_full["timestamp"] <= t_base1)
    if not mask.any():
        logging.warning("baseline_range matched no events – skipping subtraction")
        return df_analysis.copy()

    rate_bl, live_bl = rate_histogram(df_full.loc[mask], bins)

    # currently electronics vs radon use same subtraction; future hooks can differ
    if mode in ("electronics", "radon", "all"):
        net_counts = (rate_an - rate_bl) * live_time_analysis
    else:  # mode == "none"
        net_counts = rate_an * live_time_analysis

    df_out = df_analysis.copy()
    df_out["subtracted_adc_hist"] = [net_counts] * len(df_out)
    return df_out
