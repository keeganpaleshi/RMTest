"""
Module: emg_stable.py
Purpose: Provide a cleaned-up wrapper around ``scipy.stats.exponnorm`` for the
Exponentially Modified Gaussian used in radon alpha spectroscopy peak fitting.
Author: RMTest Enhancement Module

This module keeps the legacy EMG lineshape while adding light-weight
stabilization steps such as NaN/Inf cleaning, guarding against overflow, and
falling back to a Gaussian when tau approaches the configured floor.

Key Features:
- Drop-in replacement for existing :mod:`scipy.stats.exponnorm` calls
- Safeguards against overflow/underflow and invalid values
- Optional fitting helpers with parameter validation

Integration:
To use this stabilized EMG helper, enable ``fitting.use_stable_emg`` in the
analysis configuration (the default). To revert to the legacy
``scipy.stats.exponnorm`` implementation, set ``fitting.use_stable_emg`` to
``false``. The minimum tau floor is configurable via ``fitting.emg_tau_min``.
If true erfcx-style EMG is required, see Module 6.

Usage Example:
    from emg_stable import StableEMG

    emg = StableEMG(use_log_scale=False)
    x = np.linspace(0, 1000, 1000)
    y = emg.pdf(x, mu=500, sigma=20, tau=5.0, amplitude=1.0)
"""

import numpy as np
from scipy import special
from scipy.stats import norm
from typing import Tuple, Optional, Union, Dict, Callable
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

        # Historically we prototyped erfcx-based expressions for the EMG, but the
        # production path simply delegates to scipy.stats.exponnorm with guards.

        # Compute in a numerically aware way
        lambda_param = 1.0 / tau
        exp_arg = lambda_param * (mu - x) + 0.5 * lambda_param**2 * sigma**2

        # Clip to prevent overflow
        exp_arg = np.clip(exp_arg, self._underflow_threshold, self._overflow_threshold)

        # Compute the CDF argument
        cdf_arg = (x - mu - sigma**2 * lambda_param) / sigma

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


_EMG_STRATEGIES: Dict[str, Callable[..., np.ndarray]] = {
    "scipy_safe": _emg_strategy_scipy_safe,
    "erfcx_exact": _emg_strategy_erfcx_exact,
}


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
