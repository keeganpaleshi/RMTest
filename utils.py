# utils.py

import numpy as np


def to_native(obj):
    """
    Recursively convert NumPy scalar types to native Python types
    (int, float) so that JSON serialization works.
    """
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        # Convert array into list of native types
        return [to_native(x) for x in obj.tolist()]
    else:
        return obj
