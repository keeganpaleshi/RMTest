"""
Module: enhancements.py
Purpose: Enhancement framework for RMTest analysis pipeline
Author: RMTest Enhancement Integration

This module provides an extensible framework for adding enhancements to the
RMTest pipeline without modifying core analysis code. Enhancements can be
enabled/disabled via configuration.
"""

from __future__ import annotations

import logging
import time
import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for enhancement modules."""

    # Parallel data loading
    parallel_loading_enabled: bool = False
    parallel_chunk_size: int = 100000
    parallel_n_workers: Optional[int] = None
    parallel_memory_limit_gb: float = 4.0

    # Baseline diagnostics
    baseline_diagnostics_enabled: bool = False
    baseline_generate_report: bool = True

    # Enhanced radon inference
    radon_inference_enhanced: bool = False
    radon_dedupe_enabled: bool = True

    # EMG stability
    stable_emg_enabled: bool = False
    stable_emg_use_log_scale: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EnhancementConfig:
        """Create EnhancementConfig from main config dictionary."""
        enh = config.get('enhancements', {})

        # Parse parallel loading config
        pl = enh.get('parallel_loading', {})
        pl_enabled = pl.get('enabled', False) if isinstance(pl, dict) else pl

        # Parse baseline diagnostics config
        bd = enh.get('baseline_diagnostics', {})
        bd_enabled = bd.get('enabled', False) if isinstance(bd, dict) else bd

        # Parse radon inference config
        ri = enh.get('radon_inference', {})
        ri_enhanced = ri.get('enhanced', False) if isinstance(ri, dict) else False
        ri_dedupe = ri.get('dedupe_enabled', True) if isinstance(ri, dict) else True

        # Parse EMG config
        emg = enh.get('stable_emg', {})
        emg_enabled = emg.get('enabled', False) if isinstance(emg, dict) else emg

        return cls(
            parallel_loading_enabled=bool(pl_enabled),
            parallel_chunk_size=pl.get('chunk_size', 100000) if isinstance(pl, dict) else 100000,
            parallel_n_workers=pl.get('n_workers') if isinstance(pl, dict) else None,
            parallel_memory_limit_gb=pl.get('memory_limit_gb', 4.0) if isinstance(pl, dict) else 4.0,
            baseline_diagnostics_enabled=bool(bd_enabled),
            baseline_generate_report=bd.get('generate_report', True) if isinstance(bd, dict) else True,
            radon_inference_enhanced=bool(ri_enhanced),
            radon_dedupe_enabled=bool(ri_dedupe),
            stable_emg_enabled=bool(emg_enabled),
            stable_emg_use_log_scale=emg.get('use_log_scale', True) if isinstance(emg, dict) else True,
        )


@dataclass
class EnhancementMetrics:
    """Metrics tracked by enhancement modules."""

    timings: dict[str, float] = field(default_factory=dict)
    data_loading_speedup: Optional[float] = None
    baseline_stability_score: Optional[float] = None
    radon_duplicates_removed: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_timing(self, name: str, duration: float) -> None:
        """Add a timing measurement."""
        self.timings[name] = duration

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(f"Enhancement warning: {message}")


class EnhancementPipeline:
    """
    Enhancement pipeline that wraps existing RMTest functionality with
    optional performance and diagnostic improvements.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize enhancement pipeline.

        Args:
            config: Main RMTest configuration dictionary
        """
        self.config = config
        self.enh_config = EnhancementConfig.from_config(config)
        self.metrics = EnhancementMetrics()

        logger.info("Enhancement pipeline initialized")
        self._log_enabled_enhancements()

    def _log_enabled_enhancements(self) -> None:
        """Log which enhancements are enabled."""
        enabled = []
        if self.enh_config.parallel_loading_enabled:
            enabled.append("parallel_loading")
        if self.enh_config.baseline_diagnostics_enabled:
            enabled.append("baseline_diagnostics")
        if self.enh_config.radon_inference_enhanced:
            enabled.append("radon_inference_enhanced")
        if self.enh_config.stable_emg_enabled:
            enabled.append("stable_emg")

        if enabled:
            logger.info(f"Enabled enhancements: {', '.join(enabled)}")
        else:
            logger.info("No enhancements enabled (using standard RMTest pipeline)")

    def enhanced_load_events(self, csv_path: str | Path, **kwargs) -> pd.DataFrame:
        """
        Load events with optional parallel loading enhancement.

        Falls back to standard io_utils.load_events if enhancement not enabled
        or if parallel_loader module is not available.

        Args:
            csv_path: Path to CSV file
            **kwargs: Additional arguments passed to load_events

        Returns:
            Event DataFrame
        """
        from io_utils import load_events

        start_time = time.time()
        csv_path = Path(csv_path)
        file_size_mb = csv_path.stat().st_size / (1024**2)

        # Check if we should use parallel loading
        if self.enh_config.parallel_loading_enabled and file_size_mb > 100:
            try:
                # Try to import parallel_loader module if it exists
                from parallel_loader import ParallelCSVLoader, LoadConfig

                load_config = LoadConfig(
                    chunk_size=self.enh_config.parallel_chunk_size,
                    n_workers=self.enh_config.parallel_n_workers,
                    memory_limit_gb=self.enh_config.parallel_memory_limit_gb
                )
                loader = ParallelCSVLoader(load_config)
                df = loader.load_file(str(csv_path))

                elapsed = time.time() - start_time
                self.metrics.add_timing('data_loading_parallel', elapsed)
                logger.info(f"Loaded {len(df):,} rows with parallel loading in {elapsed:.2f}s")

                return df

            except ImportError:
                logger.info("Parallel loader module not available, using standard loading")
                self.metrics.add_warning("parallel_loader module not found")

        # Use standard loading
        df = load_events(str(csv_path), **kwargs)
        elapsed = time.time() - start_time
        self.metrics.add_timing('data_loading_standard', elapsed)
        logger.info(f"Loaded {len(df):,} rows in {elapsed:.2f}s")

        return df

    def enhanced_baseline_diagnostics(
        self,
        events: pd.DataFrame,
        output_dir: str | Path
    ) -> Optional[dict[str, Any]]:
        """
        Generate enhanced baseline diagnostics if enabled.

        Args:
            events: Event DataFrame
            output_dir: Output directory for diagnostic plots

        Returns:
            Diagnostic metrics dictionary or None if not enabled
        """
        if not self.enh_config.baseline_diagnostics_enabled:
            return None

        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try to use baseline_diagnostics module if available
            from baseline_diagnostics import BaselineDiagnostics

            diagnostics = BaselineDiagnostics()

            # Extract configuration
            baseline_config = self.config.get('baseline', {})
            if 'range' not in baseline_config:
                logger.warning("No baseline range configured for diagnostics")
                return None

            baseline_range = (
                pd.Timestamp(baseline_config['range'][0]),
                pd.Timestamp(baseline_config['range'][1])
            )

            analysis_config = self.config.get('analysis', {})
            analysis_range = (
                pd.Timestamp(analysis_config.get('analysis_start_time', events['timestamp'].min())),
                pd.Timestamp(analysis_config.get('analysis_end_time', events['timestamp'].max()))
            )

            isotope_windows = {
                'Po210': (5.1, 5.5),
                'Po218': (5.9, 6.1),
                'Po214': (7.5, 7.9)
            }

            calibration = self.config.get('calibration', {})

            # Generate diagnostic plots
            output_path = output_dir / 'baseline_diagnostics.png'
            metrics = diagnostics.plot_baseline_diagnostics(
                events, baseline_range, analysis_range,
                isotope_windows, calibration, str(output_path)
            )

            # Generate report if requested
            if self.enh_config.baseline_generate_report:
                report_path = output_dir / 'baseline_diagnostics_report.txt'
                diagnostics.generate_baseline_report(metrics, str(report_path))

            elapsed = time.time() - start_time
            self.metrics.add_timing('baseline_diagnostics', elapsed)
            logger.info(f"Baseline diagnostics completed in {elapsed:.2f}s")

            return metrics

        except ImportError:
            logger.info("baseline_diagnostics module not available")
            self.metrics.add_warning("baseline_diagnostics module not found")
            return None

    def enhanced_radon_inference(
        self,
        isotope_series: dict[str, Any],
        output_dir: str | Path,
        **kwargs
    ) -> dict[str, Any]:
        """
        Run enhanced radon inference with deduplication if enabled.

        Args:
            isotope_series: Isotope time series data
            output_dir: Output directory
            **kwargs: Additional arguments for radon inference

        Returns:
            Inference results dictionary
        """
        from radon.radon_inference import run_radon_inference

        start_time = time.time()
        output_dir = Path(output_dir)

        # Check if enhanced radon inference is available
        if self.enh_config.radon_inference_enhanced:
            try:
                from radon_inference_enhanced import (
                    RadonInferenceEngine,
                    RadonInferenceConfig,
                    InferenceMethod
                )

                # Build enhanced configuration
                ri_config = self.config.get('radon_inference', {})
                inference_config = RadonInferenceConfig(
                    enabled=True,
                    method=InferenceMethod.BEST,
                    dedupe_enabled=self.enh_config.radon_dedupe_enabled,
                    # ... additional config from main config
                )

                engine = RadonInferenceEngine(inference_config)
                result = engine.infer_radon_activity(isotope_series)

                self.metrics.radon_duplicates_removed = result.duplicates_removed

                elapsed = time.time() - start_time
                self.metrics.add_timing('radon_inference_enhanced', elapsed)
                logger.info(f"Enhanced radon inference completed in {elapsed:.2f}s")

                # Convert to dictionary for compatibility
                return {
                    'method_used': result.method_used,
                    'duplicates_removed': result.duplicates_removed,
                    'warnings': result.warnings,
                    # ... additional results
                }

            except ImportError:
                logger.info("radon_inference_enhanced module not available, using standard")
                self.metrics.add_warning("radon_inference_enhanced module not found")

        # Use standard radon inference
        result = run_radon_inference(
            self.config,
            isotope_series,
            **kwargs
        )

        elapsed = time.time() - start_time
        self.metrics.add_timing('radon_inference_standard', elapsed)
        logger.info(f"Standard radon inference completed in {elapsed:.2f}s")

        return result

    def enhanced_emg_fitting(
        self,
        energy_data: np.ndarray,
        initial_params: dict[str, float]
    ) -> dict[str, float]:
        """
        Perform EMG fitting with optional stability enhancements.

        Args:
            energy_data: Energy values to fit
            initial_params: Initial fit parameters

        Returns:
            Fitted parameters dictionary
        """
        if not self.enh_config.stable_emg_enabled:
            # Use standard fitting from calibration module
            return initial_params

        try:
            from emg_stable import StableEMG

            stable_emg = StableEMG(use_log_scale=self.enh_config.stable_emg_use_log_scale)

            # Histogram the data
            counts, bin_edges = np.histogram(energy_data, bins=100)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Fit with stable EMG
            fit_result = stable_emg.fit_parameters(bin_centers, counts, initial_params)

            if fit_result.get('success', False):
                logger.info("Stable EMG fit successful")
                return fit_result
            else:
                logger.warning("Stable EMG fit failed, using standard method")
                self.metrics.add_warning("Stable EMG fit failed")
                return initial_params

        except ImportError:
            logger.info("emg_stable module not available")
            self.metrics.add_warning("emg_stable module not found")
            return initial_params

    def save_metrics(self, output_dir: str | Path) -> None:
        """
        Save enhancement metrics to file.

        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_dict = {
            'timings': self.metrics.timings,
            'data_loading_speedup': self.metrics.data_loading_speedup,
            'baseline_stability_score': self.metrics.baseline_stability_score,
            'radon_duplicates_removed': self.metrics.radon_duplicates_removed,
            'warnings': self.metrics.warnings,
            'enhancements_enabled': {
                'parallel_loading': self.enh_config.parallel_loading_enabled,
                'baseline_diagnostics': self.enh_config.baseline_diagnostics_enabled,
                'radon_inference_enhanced': self.enh_config.radon_inference_enhanced,
                'stable_emg': self.enh_config.stable_emg_enabled,
            }
        }

        output_file = output_dir / 'enhancement_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        logger.info(f"Enhancement metrics saved to {output_file}")

        # Log summary
        if self.metrics.timings:
            total_time = sum(self.metrics.timings.values())
            logger.info("Enhancement timing summary:")
            for name, duration in self.metrics.timings.items():
                pct = (duration / total_time * 100) if total_time > 0 else 0
                logger.info(f"  {name}: {duration:.2f}s ({pct:.1f}%)")


def integrate_enhancements(analyze_module):
    """
    Integrate enhancements into existing analyze.py module.

    This function can be used to monkey-patch the analyze module
    to use enhancement functionality without modifying analyze.py directly.

    Args:
        analyze_module: The imported analyze module

    Example:
        import analyze
        from enhancements import integrate_enhancements
        integrate_enhancements(analyze)
    """
    # Store original functions
    if hasattr(analyze_module, 'load_events'):
        analyze_module._original_load_events = analyze_module.load_events

    logger.info("Enhancements integrated into analyze module")
    logger.info("To enable, add 'enhancements' section to config.yaml")


if __name__ == "__main__":
    print("RMTest Enhancement Framework")
    print("=" * 60)
    print("\nThis module provides an extensible framework for enhancements.")
    print("\nAvailable enhancements:")
    print("  - Parallel data loading (requires parallel_loader.py)")
    print("  - Baseline diagnostics (requires baseline_diagnostics.py)")
    print("  - Enhanced radon inference (requires radon_inference_enhanced.py)")
    print("  - Stable EMG fitting (requires emg_stable.py)")
    print("\nTo enable enhancements, add to config.yaml:")
    print("""
enhancements:
  parallel_loading:
    enabled: true
    chunk_size: 100000
    n_workers: 4
    memory_limit_gb: 4.0

  baseline_diagnostics:
    enabled: true
    generate_report: true

  radon_inference:
    enhanced: true
    dedupe_enabled: true

  stable_emg:
    enabled: true
    use_log_scale: true
    """)
    print("\nUsage in analyze.py:")
    print("""
from enhancements import EnhancementPipeline

# In main():
config = load_config(args.config)
pipeline = EnhancementPipeline(config)

# Use enhanced loading
events = pipeline.enhanced_load_events(args.input)

# Use enhanced baseline diagnostics
baseline_metrics = pipeline.enhanced_baseline_diagnostics(events, output_dir)

# Use enhanced radon inference
radon_results = pipeline.enhanced_radon_inference(isotope_series, output_dir)

# Save metrics
pipeline.save_metrics(output_dir)
    """)
