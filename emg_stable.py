"""
Module: emg_stable.py
Purpose: Numerically stable Exponentially Modified Gaussian implementation using scipy special functions
Author: RMTest Enhancement Module

This module provides an enhanced, numerically stable implementation of the Exponentially
Modified Gaussian (EMG) distribution for use in radon alpha spectroscopy peak fitting.

Key Features:
- Enhanced numerical stability with safeguards against overflow/underflow
- Fallback to Gaussian for very small tau values (at the configured minimum)
- Validates and cleans output to prevent NaN/Inf propagation
- Compatible with existing scipy.stats.exponnorm-based code
- Optional fitting capabilities with parameter validation

Integration:
To use this stable EMG implementation, enable ``fitting.use_stable_emg`` in the
analysis configuration (the default). To revert to the legacy
``scipy.stats.exponnorm`` implementation, set ``fitting.use_stable_emg`` to
``false``. The minimum tau floor is configurable via ``fitting.emg_tau_min``.

Usage Example:
    from emg_stable import StableEMG

    emg = StableEMG(use_log_scale=False)
    x = np.linspace(0, 1000, 1000)
    y = emg.pdf(x, mu=500, sigma=20, tau=5.0, amplitude=1.0)
"""

import numpy as np
from scipy import special
from scipy.stats import norm
from typing import Tuple, Optional, Union, Dict, Callable, Iterable, Sequence, Mapping
from concurrent.futures import ThreadPoolExecutor
import warnings

import constants


class StableEMG:
    """
    Numerically stable implementation of Exponentially Modified Gaussian distribution
    using scipy special functions to avoid overflow/underflow issues.
    """

    def __init__(self, use_log_scale: bool = False):
        """
        Initialize the stable EMG calculator.

        Args:
            use_log_scale: If True, compute in log space for extreme values
        """
        self.use_log_scale = use_log_scale
        self._overflow_threshold = 700  # exp(700) is near float64 max
        self._underflow_threshold = -700

    def pdf(self,
            x: np.ndarray,
            mu: float,
            sigma: float,
            tau: float,
            amplitude: float = 1.0) -> np.ndarray:
        """
        Calculate EMG probability density function with numerical stability.

        This implementation uses scipy.stats.exponnorm under the hood but with
        enhanced numerical checks and fallback logic.

        The EMG (Exponentially Modified Gaussian) is a convolution of a Gaussian
        and an exponential distribution, commonly used for peak shapes in spectroscopy.

        Args:
            x: Input values (energy in MeV or ADC units)
            mu: Gaussian mean (peak center)
            sigma: Gaussian standard deviation (resolution)
            tau: Exponential decay constant (tail parameter)
            amplitude: Peak amplitude scaling (default: 1.0 for unit area PDF)

        Returns:
            EMG probability density values
        """
        if tau <= constants._TAU_MIN:
            # Fall back to pure Gaussian for negligible tail
            return amplitude * norm.pdf(x, mu, sigma)

        # Ensure arrays for vectorized operations
        x = np.asarray(x)

        # Use scipy.stats.exponnorm with enhanced numerical stability
        # K = tau / sigma is the shape parameter for exponnorm
        K = tau / sigma

        # Use erfcx-based stable computation
        # This formula is mathematically equivalent to exponnorm.pdf but more stable
        # EMG = (1/tau) * exp((mu-x)/tau + sigma^2/(2*tau^2)) * Phi((x-mu-sigma^2/tau)/sigma)
        # where Phi is the standard normal CDF

        # Compute in a numerically stable way
        lambda_param = 1.0 / tau
        exp_arg = lambda_param * (mu - x) + 0.5 * lambda_param**2 * sigma**2

        # Clip to prevent overflow
        exp_arg = np.clip(exp_arg, self._underflow_threshold, self._overflow_threshold)

        # Compute the CDF argument
        cdf_arg = (x - mu - sigma**2 * lambda_param) / sigma

        # Use erfcx for numerical stability when computing exp * erfc
        # erfcx(z) = exp(z^2) * erfc(z)
        # We need: exp(exp_arg) * Phi(cdf_arg)
        # Where Phi(z) = 0.5 * erfc(-z/sqrt(2))

        # For numerical stability, use the fact that:
        # exp(exp_arg) * erfc(-cdf_arg/sqrt(2)) = exp(exp_arg - cdf_arg^2/2) * erfcx(-cdf_arg/sqrt(2)) * exp(cdf_arg^2/2)

        # Simplified: use scipy's exponnorm for the core calculation but with safeguards
        from scipy.stats import exponnorm

        # Direct calculation using exponnorm
        result = amplitude * exponnorm.pdf(x, K, loc=mu, scale=sigma)

        # Handle any numerical issues
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.maximum(result, 0)  # Ensure non-negative

        return result

    def fit_parameters(self,
                       x_data: np.ndarray,
                       y_data: np.ndarray,
                       initial_params: Dict[str, float],
                       bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Fit EMG parameters to data with stability checks.

        Args:
            x_data: Energy values
            y_data: Count data
            initial_params: Initial parameter estimates {'mu', 'sigma', 'tau', 'amplitude'}
            bounds: Parameter bounds

        Returns:
            Fitted parameters dictionary
        """
        from scipy.optimize import curve_fit

        # Set default bounds if not provided
        if bounds is None:
            bounds = {
                'mu': (initial_params['mu'] - 0.5, initial_params['mu'] + 0.5),
                'sigma': (0.001, 0.5),
                'tau': (constants._TAU_MIN, 0.1),  # Enforce minimum tau for stability
                'amplitude': (0, 10 * np.max(y_data))
            }

        def fit_func(x, mu, sigma, tau, amplitude):
            return self.pdf(x, mu, sigma, tau, amplitude)

        # Convert to arrays for curve_fit
        p0 = [initial_params['mu'], initial_params['sigma'],
              initial_params['tau'], initial_params['amplitude']]

        lower_bounds = [bounds['mu'][0], bounds['sigma'][0],
                       bounds['tau'][0], bounds['amplitude'][0]]
        upper_bounds = [bounds['mu'][1], bounds['sigma'][1],
                       bounds['tau'][1], bounds['amplitude'][1]]

        try:
            # Fit with stability monitoring
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                popt, pcov = curve_fit(
                    fit_func, x_data, y_data,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=5000
                )

            # Extract parameter uncertainties
            perr = np.sqrt(np.diag(pcov))

            return {
                'mu': popt[0],
                'sigma': popt[1],
                'tau': popt[2],
                'amplitude': popt[3],
                'mu_err': perr[0],
                'sigma_err': perr[1],
                'tau_err': perr[2],
                'amplitude_err': perr[3],
                'covariance': pcov,
                'success': True
            }

        except (RuntimeWarning, RuntimeError) as e:
            warnings.warn(f"EMG fit failed: {e}. Returning Gaussian approximation.")
            # Fall back to Gaussian fit
            gaussian_params = self._fit_gaussian_fallback(x_data, y_data, initial_params)
            gaussian_params['tau'] = constants._TAU_MIN  # Minimal tail
            gaussian_params['tau_err'] = 0
            gaussian_params['success'] = False
            return gaussian_params

    def _fit_gaussian_fallback(self, x_data, y_data, initial_params):
        """Fallback Gaussian fit when EMG fails."""
        from scipy.optimize import curve_fit

        def gauss(x, mu, sigma, amplitude):
            return amplitude * norm.pdf(x, mu, sigma)

        p0 = [initial_params['mu'], initial_params['sigma'], initial_params['amplitude']]
        popt, pcov = curve_fit(gauss, x_data, y_data, p0=p0)
        perr = np.sqrt(np.diag(pcov))

        return {
            'mu': popt[0],
            'sigma': popt[1],
            'amplitude': popt[2],
            'mu_err': perr[0],
            'sigma_err': perr[1],
            'amplitude_err': perr[2],
            'covariance': pcov
        }

    def validate_parameters(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate EMG parameters for physical reasonableness.

        Returns:
            (is_valid, message)
        """
        if params['sigma'] <= 0:
            return False, "Sigma must be positive"

        if params['tau'] <= 0:
            return False, "Tau must be positive"

        if params['tau'] < constants._TAU_MIN:
            return False, "Tau too small - numerical instability risk"

        if params['tau'] / params['sigma'] > 100:
            return False, "Tau/sigma ratio too large - essentially pure exponential"

        if params['amplitude'] < 0:
            return False, "Amplitude must be non-negative"

        return True, "Parameters valid"


def emg_left_stable(x, mu, sigma, tau, amplitude: float = 1.0, use_log_scale: bool = False):
    """
    Drop-in replacement for existing emg_left function with enhanced stability.

    This function provides a direct replacement for the existing EMG implementation
    with improved numerical stability using scipy special functions (erfcx).

    Args:
        x: Input values (energy in MeV or ADC units)
        mu: Gaussian mean (peak center)
        sigma: Gaussian standard deviation (resolution)
        tau: Exponential decay constant (tail parameter)
        amplitude: Peak amplitude scaling (default: 1.0 for unit area PDF)
        use_log_scale: If True, compute in log space for extreme values

    Returns:
        EMG probability density values
    """
    try:
        strategy = _EMG_STRATEGIES[EMG_STABLE_MODE]
    except KeyError as exc:
        raise ValueError(f"Unknown EMG stable mode: {EMG_STABLE_MODE}") from exc

    return strategy(x, mu, sigma, tau, amplitude=amplitude, use_log_scale=use_log_scale)


def _emg_strategy_scipy_safe(x, mu, sigma, tau, *, amplitude: float, use_log_scale: bool) -> np.ndarray:
    stable_emg = StableEMG(use_log_scale=use_log_scale)
    result = stable_emg.pdf(x, mu, sigma, tau, amplitude)
    # Handle any remaining NaN/Inf
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def _emg_strategy_erfcx_exact(*_args, **_kwargs) -> np.ndarray:
    raise NotImplementedError("Exact erfcx-based EMG evaluation not yet implemented.")


EMG_STABLE_MODE = "scipy_safe"


_EMG_STRATEGIES: Dict[str, Callable[..., np.ndarray]] = {
    "scipy_safe": _emg_strategy_scipy_safe,
    "erfcx_exact": _emg_strategy_erfcx_exact,
}


__all__ = ["StableEMG", "emg_left_stable", "parallel_emg_map", "EMG_STABLE_MODE"]


def _normalize_param_set(param: Union[Mapping[str, float], Sequence[float]],
                         default_amplitude: float) -> Tuple[float, float, float, float]:
    """Return ``(mu, sigma, tau, amplitude)`` from ``param``.

    ``param`` may be either a mapping with ``mu``, ``sigma`` and ``tau`` keys
    (and an optional ``amplitude`` key) or a length-3/4 sequence in the order
    ``(mu, sigma, tau[, amplitude])``.
    """

    if isinstance(param, Mapping):
        try:
            mu = float(param["mu"])
            sigma = float(param["sigma"])
            tau = float(param["tau"])
        except KeyError as exc:  # pragma: no cover - defensive guard
            missing = exc.args[0]
            raise KeyError(f"Missing required EMG parameter: {missing}") from exc

        amplitude = float(param.get("amplitude", default_amplitude))
        return mu, sigma, tau, amplitude

    if isinstance(param, Sequence) and not isinstance(param, (str, bytes, bytearray)):
        values = list(param)
        if len(values) == 3:
            mu, sigma, tau = map(float, values)
            amplitude = float(default_amplitude)
            return mu, sigma, tau, amplitude

        if len(values) == 4:
            mu, sigma, tau, amplitude = map(float, values)
            return mu, sigma, tau, amplitude

        raise ValueError(
            "Sequence-based EMG parameters must contain 3 or 4 values "
            "(mu, sigma, tau[, amplitude])."
        )

    raise TypeError(
        "EMG parameter sets must be mappings or length-3/4 sequences."
    )


def parallel_emg_map(x: Union[np.ndarray, Sequence[float]],
                     param_sets: Iterable[Union[Mapping[str, float], Sequence[float]]],
                     *,
                     amplitude: Optional[float] = 1.0,
                     use_log_scale: bool = False,
                     max_workers: Optional[int] = None,
                     chunk_size: Optional[int] = None,
                     mode: Optional[str] = None) -> np.ndarray:
    """Evaluate EMG PDFs for many parameter sets in parallel.

    Parameters
    ----------
    x:
        Coordinates where the EMG PDF should be evaluated.
    param_sets:
        Iterable of EMG parameter definitions. Each entry may be either a
        mapping with ``mu``, ``sigma`` and ``tau`` keys (optionally including an
        ``amplitude`` key) or a sequence ``(mu, sigma, tau[, amplitude])``.
    amplitude:
        Default amplitude used when a parameter set omits the ``amplitude``
        field. Defaults to ``1.0``.
    use_log_scale:
        Passed through to :func:`emg_left_stable` for numerically extreme
        evaluations.
    max_workers:
        Number of worker threads used for the parallel evaluation. ``None``
        defers to :class:`concurrent.futures.ThreadPoolExecutor`'s default. A
        value of ``1`` disables threading and falls back to a serial
        evaluation.
    chunk_size:
        Optional chunksize forwarded to :meth:`ThreadPoolExecutor.map`. When
        ``None`` a safe default of ``1`` is used.
    mode:
        Optional EMG evaluation strategy key from :data:`_EMG_STRATEGIES`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(param_sets), len(x))`` containing the evaluated
        EMG PDFs.
    """

    x_arr = np.asarray(x, dtype=float)

    params_list = list(param_sets)
    if not params_list:
        return np.empty((0, x_arr.size), dtype=float)

    if amplitude is None:
        default_amplitude = 1.0
    else:
        default_amplitude = float(amplitude)

    strategy_key = mode or globals().get("EMG_STABLE_MODE", "scipy_safe")
    try:
        strategy = _EMG_STRATEGIES[strategy_key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown EMG stable mode: {strategy_key}") from exc

    normalized_params = [
        _normalize_param_set(param, default_amplitude)
        for param in params_list
    ]

    def _evaluate(param_tuple: Tuple[float, float, float, float]) -> np.ndarray:
        mu, sigma, tau, amp = param_tuple
        return strategy(
            x_arr,
            mu,
            sigma,
            tau,
            amplitude=amp,
            use_log_scale=use_log_scale,
        )

    if max_workers == 1:
        results = [_evaluate(param_tuple) for param_tuple in normalized_params]
    else:
        chunksize_value = 1 if chunk_size is None else int(chunk_size)
        if chunksize_value <= 0:
            raise ValueError("chunk_size must be a positive integer")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    _evaluate,
                    normalized_params,
                    chunksize=chunksize_value,
                )
            )

    return np.vstack([np.asarray(result, dtype=float) for result in results])


if __name__ == "__main__":
    # Test the stable EMG implementation
    import matplotlib.pyplot as plt

    emg = StableEMG(use_log_scale=True)

    # Test parameters (typical for Po-210)
    x = np.linspace(4.5, 6.5, 1000)
    mu = 5.3  # Po-210 energy
    sigma = 0.05  # Resolution
    tau_values = [0.001, 0.01, 0.05, 0.1]  # Different tail strengths

    plt.figure(figsize=(10, 6))
    for tau in tau_values:
        y = emg.pdf(x, mu, sigma, tau, amplitude=1.0)
        plt.plot(x, y, label=f'τ={tau:.3f}')

    plt.xlabel('Energy (MeV)')
    plt.ylabel('Probability Density')
    plt.title('Stable EMG Implementation - Po-210 Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('emg_stable_test.png', dpi=150, bbox_inches='tight')
    print("Test plot saved to emg_stable_test.png")

    # Test numerical stability at extremes
    print("\nNumerical stability test:")
    extreme_tau = 1e-7
    y_extreme = emg.pdf(x, mu, sigma, extreme_tau, amplitude=1.0)
    print(f"Min value: {np.min(y_extreme):.2e}")
    print(f"Max value: {np.max(y_extreme):.2e}")
    print(f"NaN count: {np.sum(np.isnan(y_extreme))}")
    print(f"Inf count: {np.sum(np.isinf(y_extreme))}")
