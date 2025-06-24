from baseline_utils import rate_histogram, subtract_baseline_df

__all__ = ["rate_histogram", "subtract_baseline"]


def subtract_baseline(*args, **kwargs):
    """Wrapper for :func:`baseline_utils.subtract_baseline_df`."""
    return subtract_baseline_df(*args, **kwargs)
