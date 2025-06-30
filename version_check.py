from packaging.version import Version
import numpy
import scipy

def check_versions():
    np_version = Version(numpy.__version__)
    sp_version = Version(scipy.__version__)
    if np_version >= Version("2.0"):
        raise RuntimeError(
            f"NumPy {numpy.__version__} is not supported; install <2.0 for compatibility."
        )
    if sp_version >= Version("1.13"):
        raise RuntimeError(
            f"SciPy {scipy.__version__} is not supported; install <1.13 or upgrade SciPy to one built for NumPy {numpy.__version__}."
        )
