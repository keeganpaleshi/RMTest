from baseline_utils import rate_histogram, subtract


def calc_activity(counts, live_time_s):
    return counts / live_time_s  # Bq

# Backwards compatibility
subtract_baseline = subtract

__all__ = ["rate_histogram", "subtract", "subtract_baseline", "calc_activity"]
