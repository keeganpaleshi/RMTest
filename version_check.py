"""Dependency version guards used during application startup.

This module should be invoked early in the startup path (for example, as one
of the first calls in CLI entry points) so the application fails fast before
initializing heavier components.
"""

from packaging.version import Version
import numpy
import scipy

def check_versions():
    """Validate supported NumPy and SciPy versions and raise on incompatibility.

    NumPy releases 2.0 and newer are not supported, so the function raises a
    ``RuntimeError`` with the message ``"NumPy <version> is not supported;
    install <2 for compatibility."`` when a 2.x version is detected. SciPy
    releases 1.13 and newer are likewise rejected: the guard raises
    ``RuntimeError`` with the message ``"SciPy <version> is not supported;
    install <1.13 or upgrade SciPy to one built for NumPy <numpy_version>."``
    to remind users to align SciPy with the installed NumPy version.
    """
    np_version = Version(numpy.__version__)
    sp_version = Version(scipy.__version__)
    if np_version >= Version("2"):
        raise RuntimeError(
            f"NumPy {numpy.__version__} is not supported; install <2 for compatibility."
        )
    if sp_version >= Version("1.13"):
        raise RuntimeError(
            f"SciPy {scipy.__version__} is not supported; install <1.13 or upgrade SciPy to one built for NumPy {numpy.__version__}."
        )
