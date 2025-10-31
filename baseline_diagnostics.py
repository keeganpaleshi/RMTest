"""
Module: baseline_diagnostics.py
Purpose: Comprehensive baseline diagnostic plots and validation for radon monitor data
Author: RMTest Enhancement Module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
from scipy import stats
from scipy.signal import savgol_filter
import logging


@dataclass
class BaselineMetrics:
    """Container for baseline quality metrics."""
    mean_rate: float
    std_rate: float
    trend_slope: float
    trend_pvalue: float
    outlier_fraction: float
    stability_score: float
    chi2_statistic: float
    chi2_pvalue: float
    duration_hours: float
    total_counts: int
    is_stable: bool
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return asdict(self)


class BaselineDiagnostics:
    """
    Generate comprehensive diagnostic plots and metrics for baseline measurements.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the diagnostic plotter.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if specified style not available
            plt.style.use('default')
        self.logger = logging.getLogger(__name__)

        # Use seaborn colors if available, otherwise use matplotlib default
        if HAS_SEABORN:
            self.color_palette = sns.color_palette("husl", 8)
        else:
            # Fallback to matplotlib's default color cycle
            self.color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color'][:8]

    def analyze_baseline(self,
                        events: pd.DataFrame,
                        baseline_range: Tuple[pd.Timestamp, pd.Timestamp],
                        isotope_windows: Dict[str, Tuple[float, float]],
                        calibration: Dict[str, float]) -> BaselineMetrics:
        """
        Analyze baseline quality and stability.

        Args:
            events: Event DataFrame with timestamp and adc columns
            baseline_range: (start, end) timestamps for baseline period
            isotope_windows: Energy windows for each isotope
            calibration: Calibration parameters (must have 'slope_MeV_per_ch')

        Returns:
            BaselineMetrics with quality assessment
        """
        # Filter to baseline period
        baseline_mask = (events['timestamp'] >= baseline_range[0]) & \
                       (events['timestamp'] <= baseline_range[1])
        baseline_events = events[baseline_mask].copy()

        if len(baseline_events) == 0:
            raise ValueError("No events found in baseline period")

        # Apply calibration
        slope = calibration.get('slope_MeV_per_ch')
        if slope is None:
            raise ValueError("calibration must contain 'slope_MeV_per_ch'")

        intercept = calibration.get('intercept_MeV', 0)
        baseline_events['energy'] = baseline_events['adc'] * slope + intercept

        # Calculate time bins (hourly)
        duration_hours = (baseline_range[1] - baseline_range[0]).total_seconds() / 3600
        n_bins = max(int(duration_hours), 1)

        baseline_events['time_bin'] = pd.cut(
            baseline_events['timestamp'],
            bins=n_bins,
            labels=False
        )

        # Count rates per bin
        bin_counts = baseline_events.groupby('time_bin').size()
        bin_duration = duration_hours / n_bins
        bin_rates = bin_counts / (bin_duration * 3600)  # counts per second

        # Calculate metrics
        mean_rate = bin_rates.mean()
        std_rate = bin_rates.std()

        # Trend analysis
        x = np.arange(len(bin_rates))
        if len(x) > 1:
            slope_val, intercept, r_value, p_value, std_err = stats.linregress(x, bin_rates)
            trend_slope = slope_val
            trend_pvalue = p_value
        else:
            trend_slope = 0
            trend_pvalue = 1

        # Outlier detection (Modified Z-score)
        median_rate = bin_rates.median()
        mad = np.median(np.abs(bin_rates - median_rate))
        modified_z_scores = 0.6745 * (bin_rates - median_rate) / (mad + 1e-10)
        outliers = np.abs(modified_z_scores) > 3.5
        outlier_fraction = outliers.sum() / len(outliers) if len(outliers) > 0 else 0

        # Chi-squared test for consistency
        expected_counts = np.full(len(bin_counts), bin_counts.mean())
        if np.all(expected_counts > 0):
            chi2_stat, chi2_p = stats.chisquare(bin_counts, expected_counts)
        else:
            chi2_stat, chi2_p = 0, 1

        # Stability score (0-1, higher is better)
        cv = std_rate / (mean_rate + 1e-10)  # Coefficient of variation
        stability_score = np.exp(-cv)  # Exponential decay from CV

        # Determine if baseline is stable
        warnings_list = []
        is_stable = True

        if cv > 0.1:
            warnings_list.append(f"High variability: CV={cv:.3f}")
            is_stable = False

        if trend_pvalue < 0.05 and abs(trend_slope) > 0.01 * mean_rate:
            warnings_list.append(f"Significant trend detected: slope={trend_slope:.3e}")
            is_stable = False

        if outlier_fraction > 0.1:
            warnings_list.append(f"Many outliers: {outlier_fraction:.1%}")
            is_stable = False

        if chi2_p < 0.01:
            warnings_list.append(f"Non-Poisson behavior: χ²_p={chi2_p:.3f}")

        if duration_hours < 24:
            warnings_list.append(f"Short baseline: {duration_hours:.1f} hours")

        return BaselineMetrics(
            mean_rate=mean_rate,
            std_rate=std_rate,
            trend_slope=trend_slope,
            trend_pvalue=trend_pvalue,
            outlier_fraction=outlier_fraction,
            stability_score=stability_score,
            chi2_statistic=chi2_stat,
            chi2_pvalue=chi2_p,
            duration_hours=duration_hours,
            total_counts=len(baseline_events),
            is_stable=is_stable,
            warnings=warnings_list
        )

    def plot_baseline_diagnostics(self,
                                  events: pd.DataFrame,
                                  baseline_range: Tuple[pd.Timestamp, pd.Timestamp],
                                  analysis_range: Tuple[pd.Timestamp, pd.Timestamp],
                                  isotope_windows: Dict[str, Tuple[float, float]],
                                  calibration: Dict[str, float],
                                  output_path: str = 'baseline_diagnostics.png') -> BaselineMetrics:
        """
        Generate comprehensive baseline diagnostic plot.

        Args:
            events: Complete event DataFrame
            baseline_range: Baseline period
            analysis_range: Analysis period for comparison
            isotope_windows: Energy windows for isotopes
            calibration: Calibration parameters
            output_path: Where to save the plot

        Returns:
            BaselineMetrics from the analysis
        """
        # Analyze baseline
        metrics = self.analyze_baseline(events, baseline_range, isotope_windows, calibration)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Time series of count rates
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_rate_timeline(ax1, events, baseline_range, analysis_range, calibration)

        # 2. Energy spectrum comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_spectrum_comparison(ax2, events, baseline_range, analysis_range,
                                      calibration, isotope_windows)

        # 3. Rate distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_rate_distribution(ax3, events, baseline_range, calibration)

        # 4. Stability metrics
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_stability_metrics(ax4, metrics)

        # 5. Isotope-specific rates
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_isotope_rates(ax5, events, baseline_range, isotope_windows, calibration)

        # 6. Autocorrelation
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_autocorrelation(ax6, events, baseline_range, calibration)

        # 7. QQ plot for Poisson check
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_qq_poisson(ax7, events, baseline_range, calibration)

        # Add overall title with status
        status = "✓ STABLE" if metrics.is_stable else "⚠ UNSTABLE"
        color = 'green' if metrics.is_stable else 'orange'
        fig.suptitle(f'Baseline Diagnostics - {status}', fontsize=16, color=color, fontweight='bold')

        # Add warnings text if any
        if metrics.warnings:
            warning_text = "Warnings: " + " | ".join(metrics.warnings)
            fig.text(0.5, 0.01, warning_text, ha='center', fontsize=10, color='red',
                    wrap=True)

        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Diagnostic plot saved to {output_path}")

        return metrics

    def _plot_rate_timeline(self, ax, events, baseline_range, analysis_range, calibration):
        """Plot count rate over time."""
        # Bin events by hour
        all_events = events.copy()
        slope = calibration.get('slope_MeV_per_ch', 1.0)
        intercept = calibration.get('intercept_MeV', 0)
        all_events['energy'] = all_events['adc'] * slope + intercept

        # Create hourly bins
        time_min = min(baseline_range[0], analysis_range[0])
        time_max = max(baseline_range[1], analysis_range[1])
        bins = pd.date_range(time_min, time_max, freq='H')

        if len(bins) < 2:
            ax.text(0.5, 0.5, 'Insufficient time range', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Count Rate Timeline')
            return

        all_events['hour_bin'] = pd.cut(all_events['timestamp'], bins=bins)
        hourly_counts = all_events.groupby('hour_bin').size()

        # Plot
        x = hourly_counts.index.map(lambda x: x.left).to_numpy()
        y = hourly_counts.values / 3600  # Convert to rate

        ax.plot(x, y, 'b-', alpha=0.7, label='Hourly rate')

        # Highlight baseline period
        ax.axvspan(baseline_range[0], baseline_range[1], alpha=0.2, color='green', label='Baseline')
        ax.axvspan(analysis_range[0], analysis_range[1], alpha=0.2, color='blue', label='Analysis')

        # Add smoothed trend
        if len(y) > 5:
            window = min(5, len(y) if len(y) % 2 == 1 else len(y)-1)
            if window >= 3:
                y_smooth = savgol_filter(y, window, 2)
                ax.plot(x, y_smooth, 'r-', linewidth=2, alpha=0.8, label='Smoothed')

        ax.set_xlabel('Time')
        ax.set_ylabel('Count Rate (cps)')
        ax.set_title('Count Rate Timeline')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_spectrum_comparison(self, ax, events, baseline_range, analysis_range,
                                 calibration, isotope_windows):
        """Compare energy spectra."""
        slope = calibration.get('slope_MeV_per_ch', 1.0)
        intercept = calibration.get('intercept_MeV', 0)

        # Get baseline events
        baseline_mask = (events['timestamp'] >= baseline_range[0]) & \
                       (events['timestamp'] <= baseline_range[1])
        baseline_events = events[baseline_mask].copy()
        baseline_events['energy'] = baseline_events['adc'] * slope + intercept

        # Get analysis events
        analysis_mask = (events['timestamp'] >= analysis_range[0]) & \
                       (events['timestamp'] <= analysis_range[1])
        analysis_events = events[analysis_mask].copy()
        analysis_events['energy'] = analysis_events['adc'] * slope + intercept

        # Plot histograms
        bins = np.linspace(0, 10, 100)

        ax.hist(baseline_events['energy'], bins=bins, alpha=0.5,
               label='Baseline', color='green', density=True)
        ax.hist(analysis_events['energy'], bins=bins, alpha=0.5,
               label='Analysis', color='blue', density=True)

        # Mark isotope windows
        colors = {'Po210': 'red', 'Po214': 'orange', 'Po218': 'purple'}
        for isotope, (e_min, e_max) in isotope_windows.items():
            ax.axvspan(e_min, e_max, alpha=0.2, color=colors.get(isotope, 'gray'))
            ax.text((e_min + e_max)/2, ax.get_ylim()[1]*0.9, isotope,
                   ha='center', fontsize=8)

        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Normalized Counts')
        ax.set_title('Energy Spectrum Comparison')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 10)

    def _plot_rate_distribution(self, ax, events, baseline_range, calibration):
        """Plot distribution of count rates."""
        baseline_mask = (events['timestamp'] >= baseline_range[0]) & \
                       (events['timestamp'] <= baseline_range[1])
        baseline_events = events[baseline_mask]

        # Bin by 10-minute intervals
        duration = (baseline_range[1] - baseline_range[0]).total_seconds()
        n_bins = max(int(duration / 600), 1)  # 10-minute bins

        time_bins = pd.cut(baseline_events['timestamp'], bins=n_bins)
        bin_counts = baseline_events.groupby(time_bins).size()
        rates = bin_counts / 600  # cps

        if len(rates) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Rate Distribution')
            return

        # Plot histogram
        ax.hist(rates, bins=20, alpha=0.7, color='green', edgecolor='black')

        # Add Poisson fit
        mean_rate = rates.mean()
        x_fit = np.linspace(rates.min(), rates.max(), 100)
        poisson_fit = stats.poisson.pmf(np.round(x_fit * 600), mean_rate * 600) * 600
        ax.plot(x_fit, poisson_fit * len(rates) * (rates.max() - rates.min()) / 20,
               'r-', linewidth=2, label=f'Poisson (λ={mean_rate:.3f})')

        ax.set_xlabel('Count Rate (cps)')
        ax.set_ylabel('Frequency')
        ax.set_title('Rate Distribution (10-min bins)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_stability_metrics(self, ax, metrics):
        """Plot stability metrics as a dashboard."""
        ax.axis('off')

        # Create metric cards
        y_positions = [0.8, 0.6, 0.4, 0.2]
        metric_data = [
            ('Stability Score', f'{metrics.stability_score:.2f}',
             'green' if metrics.stability_score > 0.9 else 'orange'),
            ('Mean Rate', f'{metrics.mean_rate:.3f} cps', 'black'),
            ('CV', f'{metrics.std_rate/metrics.mean_rate:.3f}',
             'green' if metrics.std_rate/metrics.mean_rate < 0.1 else 'orange'),
            ('Duration', f'{metrics.duration_hours:.1f} hours',
             'green' if metrics.duration_hours > 24 else 'orange')
        ]

        for y, (label, value, color) in zip(y_positions, metric_data):
            ax.text(0.1, y, label + ':', fontsize=12, fontweight='bold')
            ax.text(0.6, y, value, fontsize=12, color=color)

        # Add status box
        status_color = 'lightgreen' if metrics.is_stable else 'lightyellow'
        rect = Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2,
                        edgecolor='black', facecolor=status_color, alpha=0.3)
        ax.add_patch(rect)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Stability Metrics', fontsize=12, fontweight='bold')

    def _plot_isotope_rates(self, ax, events, baseline_range, isotope_windows, calibration):
        """Plot isotope-specific count rates."""
        baseline_mask = (events['timestamp'] >= baseline_range[0]) & \
                       (events['timestamp'] <= baseline_range[1])
        baseline_events = events[baseline_mask].copy()

        slope = calibration.get('slope_MeV_per_ch', 1.0)
        intercept = calibration.get('intercept_MeV', 0)
        baseline_events['energy'] = baseline_events['adc'] * slope + intercept

        isotope_counts = {}
        for isotope, (e_min, e_max) in isotope_windows.items():
            mask = (baseline_events['energy'] >= e_min) & (baseline_events['energy'] <= e_max)
            isotope_counts[isotope] = mask.sum()

        duration_hours = (baseline_range[1] - baseline_range[0]).total_seconds() / 3600

        # Plot bar chart
        isotopes = list(isotope_counts.keys())
        rates = [count / (duration_hours * 3600) for count in isotope_counts.values()]

        bars = ax.bar(isotopes, rates, color=['red', 'orange', 'purple'][:len(isotopes)])

        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.4f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Count Rate (cps)')
        ax.set_title('Isotope-Specific Rates')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_autocorrelation(self, ax, events, baseline_range, calibration):
        """Plot autocorrelation of count rates."""
        baseline_mask = (events['timestamp'] >= baseline_range[0]) & \
                       (events['timestamp'] <= baseline_range[1])
        baseline_events = events[baseline_mask]

        # Bin by minutes
        duration = (baseline_range[1] - baseline_range[0]).total_seconds()
        n_bins = max(int(duration / 60), 10)

        time_bins = pd.cut(baseline_events['timestamp'], bins=n_bins)
        bin_counts = baseline_events.groupby(time_bins).size().values

        # Calculate autocorrelation
        if len(bin_counts) > 10:
            acf = np.correlate(bin_counts - bin_counts.mean(),
                              bin_counts - bin_counts.mean(), mode='full')
            acf = acf[len(acf)//2:] / acf[len(acf)//2]
            lags = np.arange(min(50, len(acf)))

            ax.stem(lags, acf[:len(lags)], basefmt=' ')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(1.96/np.sqrt(len(bin_counts)), color='red',
                      linestyle='--', alpha=0.5, label='95% CI')
            ax.axhline(-1.96/np.sqrt(len(bin_counts)), color='red',
                      linestyle='--', alpha=0.5)

            ax.set_xlabel('Lag (minutes)')
            ax.set_ylabel('Autocorrelation')
            ax.set_title('Autocorrelation Function')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Autocorrelation Function')

    def _plot_qq_poisson(self, ax, events, baseline_range, calibration):
        """QQ plot to check Poisson distribution."""
        baseline_mask = (events['timestamp'] >= baseline_range[0]) & \
                       (events['timestamp'] <= baseline_range[1])
        baseline_events = events[baseline_mask]

        # Bin by 10 minutes
        duration = (baseline_range[1] - baseline_range[0]).total_seconds()
        n_bins = max(int(duration / 600), 5)

        time_bins = pd.cut(baseline_events['timestamp'], bins=n_bins)
        bin_counts = baseline_events.groupby(time_bins).size().values

        if len(bin_counts) > 3:
            # Calculate theoretical quantiles for Poisson
            mean_count = bin_counts.mean()
            theoretical = stats.poisson.ppf(
                np.linspace(0.01, 0.99, len(bin_counts)),
                mean_count
            )

            # Sort observed counts
            observed = np.sort(bin_counts)

            ax.scatter(theoretical, observed, alpha=0.6)
            ax.plot([min(theoretical), max(theoretical)],
                   [min(theoretical), max(theoretical)],
                   'r--', label='Perfect Poisson')

            ax.set_xlabel('Theoretical Quantiles (Poisson)')
            ax.set_ylabel('Observed Quantiles')
            ax.set_title('QQ Plot - Poisson Check')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('QQ Plot - Poisson Check')

    def generate_baseline_report(self,
                                 metrics: BaselineMetrics,
                                 output_path: str = 'baseline_report.txt'):
        """
        Generate a text report of baseline quality.

        Args:
            metrics: Baseline metrics from analysis
            output_path: Where to save the report
        """
        report_lines = [
            "=" * 60,
            "BASELINE QUALITY REPORT",
            "=" * 60,
            "",
            f"Duration: {metrics.duration_hours:.2f} hours",
            f"Total Events: {metrics.total_counts:,}",
            "",
            "STABILITY METRICS:",
            f"  Mean Rate: {metrics.mean_rate:.4f} ± {metrics.std_rate:.4f} cps",
            f"  Coefficient of Variation: {metrics.std_rate/metrics.mean_rate:.3f}",
            f"  Stability Score: {metrics.stability_score:.3f} (0-1 scale)",
            "",
            "STATISTICAL TESTS:",
            f"  Trend Test: slope={metrics.trend_slope:.3e}, p={metrics.trend_pvalue:.3f}",
            f"  Chi-squared Test: χ²={metrics.chi2_statistic:.2f}, p={metrics.chi2_pvalue:.3f}",
            f"  Outlier Fraction: {metrics.outlier_fraction:.1%}",
            "",
            f"OVERALL STATUS: {'STABLE' if metrics.is_stable else 'UNSTABLE'}",
            ""
        ]

        if metrics.warnings:
            report_lines.extend([
                "WARNINGS:",
                *[f"  - {warning}" for warning in metrics.warnings],
                ""
            ])

        report_lines.extend([
            "RECOMMENDATIONS:",
        ])

        if metrics.duration_hours < 24:
            report_lines.append("  - Consider longer baseline (>24 hours)")

        if metrics.stability_score < 0.9:
            report_lines.append("  - Investigate source of instability")

        if metrics.outlier_fraction > 0.05:
            report_lines.append("  - Check for electronic noise or interference")

        report_lines.extend(["", "=" * 60])

        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Report saved to {output_path}")

        # Also print to console
        print('\n'.join(report_lines))


def build_baseline_diagnostics(
    events: pd.DataFrame,
    baseline_info: Dict[str, Any],
    calibration: Dict[str, float],
    isotope_windows: Optional[Dict[str, Tuple[float, float]]] = None,
    output_dir: str = '.'
) -> Dict[str, Any]:
    """
    Build baseline diagnostics for inclusion in reporting.

    Args:
        events: Event DataFrame with timestamp and adc columns
        baseline_info: Baseline info dict with 'start' and 'end' timestamps
        calibration: Calibration parameters
        isotope_windows: Optional isotope windows, defaults to standard Po windows
        output_dir: Directory for output files

    Returns:
        Dict with baseline diagnostic metrics
    """
    logger = logging.getLogger(__name__)

    # Default isotope windows if not provided
    if isotope_windows is None:
        isotope_windows = {
            'Po210': (5.1, 5.5),
            'Po214': (7.5, 7.9),
            'Po218': (5.9, 6.1)
        }

    # Extract baseline range
    baseline_start = baseline_info.get('start')
    baseline_end = baseline_info.get('end')

    if baseline_start is None or baseline_end is None:
        logger.warning("Baseline diagnostics skipped: no baseline time range defined")
        return {}

    # Ensure timestamps are pandas Timestamps
    if not isinstance(baseline_start, pd.Timestamp):
        baseline_start = pd.Timestamp(baseline_start)
    if not isinstance(baseline_end, pd.Timestamp):
        baseline_end = pd.Timestamp(baseline_end)

    baseline_range = (baseline_start, baseline_end)

    # Use full data range as analysis range for comparison
    analysis_range = (events['timestamp'].min(), events['timestamp'].max())

    try:
        # Create diagnostics object
        diagnostics = BaselineDiagnostics()

        # Generate diagnostic plot
        import os
        plot_path = os.path.join(output_dir, 'baseline_diagnostics.png')
        metrics = diagnostics.plot_baseline_diagnostics(
            events=events,
            baseline_range=baseline_range,
            analysis_range=analysis_range,
            isotope_windows=isotope_windows,
            calibration=calibration,
            output_path=plot_path
        )

        # Generate text report
        report_path = os.path.join(output_dir, 'baseline_quality_report.txt')
        diagnostics.generate_baseline_report(metrics, output_path=report_path)

        # Return metrics as dict
        result = metrics.to_dict()
        result['diagnostic_plot'] = plot_path
        result['diagnostic_report'] = report_path

        return result

    except Exception as e:
        logger.error(f"Error generating baseline diagnostics: {e}", exc_info=True)
        return {'error': str(e)}


__all__ = [
    "BaselineMetrics",
    "BaselineDiagnostics",
    "build_baseline_diagnostics",
]


if __name__ == "__main__":
    print("Baseline Diagnostics Module")
    print("=" * 40)
    print("This module provides:")
    print("- Comprehensive baseline stability analysis")
    print("- Multi-panel diagnostic plots")
    print("- Statistical tests for data quality")
    print("- Automated report generation")
    print("\nExample isotope windows (MeV):")
    print("  Po-210: 5.1-5.5")
    print("  Po-214: 7.5-7.9")
    print("  Po-218: 5.9-6.1")
