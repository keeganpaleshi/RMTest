import numpy as np

__all__ = ["subtract_baseline"]


def subtract_baseline(counts, efficiency, live_time, baseline_counts, baseline_live_time):
    rate = counts / live_time / efficiency
    sigma_sq = counts / live_time**2 / efficiency**2
    baseline_rate = baseline_counts / baseline_live_time / efficiency
    baseline_sigma_sq = baseline_counts / baseline_live_time**2 / efficiency**2
    corrected_rate = rate - baseline_rate
    corrected_sigma = np.sqrt(sigma_sq + baseline_sigma_sq)
    return corrected_rate, corrected_sigma
