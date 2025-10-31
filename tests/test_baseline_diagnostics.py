"""Tests for baseline_diagnostics module."""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path

from baseline_diagnostics import (
    BaselineMetrics,
    BaselineDiagnostics,
    build_baseline_diagnostics,
)


@pytest.fixture
def sample_events():
    """Create sample event data for testing."""
    np.random.seed(42)

    # Generate 1000 events over 48 hours
    n_events = 1000
    start_time = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')

    # Generate timestamps with some structure
    timestamps = pd.date_range(
        start=start_time,
        periods=n_events,
        freq='173s'  # ~1000 events in 48 hours
    )

    # Generate ADC values centered around Po-214 peak (7.7 MeV)
    # Assume calibration of 0.01 MeV/channel
    adc_values = np.random.normal(770, 50, n_events)

    # Add some Po-218 events (6.0 MeV)
    n_po218 = 200
    adc_values[:n_po218] = np.random.normal(600, 30, n_po218)

    # Add some Po-210 events (5.3 MeV)
    n_po210 = 100
    adc_values[n_po218:n_po218+n_po210] = np.random.normal(530, 20, n_po210)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'adc': adc_values
    })

    return df


@pytest.fixture
def calibration():
    """Create sample calibration parameters."""
    return {
        'slope_MeV_per_ch': 0.01,
        'intercept_MeV': 0.0
    }


@pytest.fixture
def isotope_windows():
    """Create standard isotope windows."""
    return {
        'Po210': (5.1, 5.5),
        'Po214': (7.5, 7.9),
        'Po218': (5.9, 6.1)
    }


@pytest.fixture
def baseline_range(sample_events):
    """Create baseline time range (first 24 hours)."""
    start = sample_events['timestamp'].min()
    end = start + pd.Timedelta(hours=24)
    return (start, end)


@pytest.fixture
def analysis_range(sample_events):
    """Create analysis time range (last 24 hours)."""
    end = sample_events['timestamp'].max()
    start = end - pd.Timedelta(hours=24)
    return (start, end)


class TestBaselineMetrics:
    """Test the BaselineMetrics dataclass."""

    def test_baseline_metrics_creation(self):
        """Test creating BaselineMetrics."""
        metrics = BaselineMetrics(
            mean_rate=0.5,
            std_rate=0.05,
            trend_slope=0.001,
            trend_pvalue=0.1,
            outlier_fraction=0.05,
            stability_score=0.95,
            chi2_statistic=10.5,
            chi2_pvalue=0.3,
            duration_hours=24.0,
            total_counts=1000,
            is_stable=True,
            warnings=[]
        )

        assert metrics.mean_rate == 0.5
        assert metrics.is_stable is True
        assert len(metrics.warnings) == 0

    def test_baseline_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = BaselineMetrics(
            mean_rate=0.5,
            std_rate=0.05,
            trend_slope=0.001,
            trend_pvalue=0.1,
            outlier_fraction=0.05,
            stability_score=0.95,
            chi2_statistic=10.5,
            chi2_pvalue=0.3,
            duration_hours=24.0,
            total_counts=1000,
            is_stable=True,
            warnings=['test warning']
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict['mean_rate'] == 0.5
        assert metrics_dict['is_stable'] is True
        assert 'test warning' in metrics_dict['warnings']


class TestBaselineDiagnostics:
    """Test the BaselineDiagnostics class."""

    def test_initialization(self):
        """Test initializing BaselineDiagnostics."""
        diag = BaselineDiagnostics()
        assert diag.logger is not None
        assert len(diag.color_palette) > 0

    def test_analyze_baseline_basic(self, sample_events, baseline_range,
                                    isotope_windows, calibration):
        """Test basic baseline analysis."""
        diag = BaselineDiagnostics()

        metrics = diag.analyze_baseline(
            events=sample_events,
            baseline_range=baseline_range,
            isotope_windows=isotope_windows,
            calibration=calibration
        )

        assert isinstance(metrics, BaselineMetrics)
        assert metrics.mean_rate > 0
        assert metrics.std_rate >= 0
        assert metrics.duration_hours > 0
        assert metrics.total_counts > 0
        assert isinstance(metrics.is_stable, bool)
        assert isinstance(metrics.warnings, list)

    def test_analyze_baseline_no_events(self, sample_events, isotope_windows, calibration):
        """Test baseline analysis with no events in range."""
        diag = BaselineDiagnostics()

        # Use a range with no events
        start = sample_events['timestamp'].min() - pd.Timedelta(days=1)
        end = sample_events['timestamp'].min() - pd.Timedelta(hours=1)

        with pytest.raises(ValueError, match="No events found in baseline period"):
            diag.analyze_baseline(
                events=sample_events,
                baseline_range=(start, end),
                isotope_windows=isotope_windows,
                calibration=calibration
            )

    def test_analyze_baseline_missing_calibration(self, sample_events, baseline_range, isotope_windows):
        """Test baseline analysis with missing calibration."""
        diag = BaselineDiagnostics()

        bad_calibration = {'intercept_MeV': 0.0}  # Missing slope

        with pytest.raises(ValueError, match="slope_MeV_per_ch"):
            diag.analyze_baseline(
                events=sample_events,
                baseline_range=baseline_range,
                isotope_windows=isotope_windows,
                calibration=bad_calibration
            )

    def test_plot_baseline_diagnostics(self, sample_events, baseline_range,
                                       analysis_range, isotope_windows, calibration):
        """Test generating baseline diagnostic plots."""
        diag = BaselineDiagnostics()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_diagnostics.png')

            metrics = diag.plot_baseline_diagnostics(
                events=sample_events,
                baseline_range=baseline_range,
                analysis_range=analysis_range,
                isotope_windows=isotope_windows,
                calibration=calibration,
                output_path=output_path
            )

            assert isinstance(metrics, BaselineMetrics)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_generate_baseline_report(self, sample_events, baseline_range,
                                      isotope_windows, calibration):
        """Test generating baseline text report."""
        diag = BaselineDiagnostics()

        metrics = diag.analyze_baseline(
            events=sample_events,
            baseline_range=baseline_range,
            isotope_windows=isotope_windows,
            calibration=calibration
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, 'test_report.txt')

            diag.generate_baseline_report(metrics, output_path=report_path)

            assert os.path.exists(report_path)

            # Check report contents
            with open(report_path, 'r') as f:
                content = f.read()

            assert 'BASELINE QUALITY REPORT' in content
            assert 'Duration:' in content
            assert 'Total Events:' in content
            assert 'STABILITY METRICS:' in content
            assert 'STATISTICAL TESTS:' in content

    def test_stability_detection_stable(self, sample_events, baseline_range,
                                       isotope_windows, calibration):
        """Test that stable baseline is detected as stable."""
        diag = BaselineDiagnostics()

        # Create very stable data (constant rate)
        n_events = 1000
        timestamps = pd.date_range(
            start=baseline_range[0],
            end=baseline_range[1],
            periods=n_events
        )

        stable_events = pd.DataFrame({
            'timestamp': timestamps,
            'adc': np.full(n_events, 770.0)  # Constant ADC
        })

        metrics = diag.analyze_baseline(
            events=stable_events,
            baseline_range=baseline_range,
            isotope_windows=isotope_windows,
            calibration=calibration
        )

        # Stable data should have high stability score
        assert metrics.stability_score > 0.8
        assert metrics.outlier_fraction < 0.15

    def test_stability_detection_unstable(self, sample_events, baseline_range,
                                         isotope_windows, calibration):
        """Test that unstable baseline is detected as unstable."""
        diag = BaselineDiagnostics()

        # Create unstable data with strong trend
        n_events = 1000
        timestamps = pd.date_range(
            start=baseline_range[0],
            end=baseline_range[1],
            periods=n_events
        )

        # Add strong linear trend
        time_indices = np.arange(n_events)
        trend = time_indices * 0.5  # Strong positive trend

        unstable_events = pd.DataFrame({
            'timestamp': timestamps,
            'adc': 770.0 + trend
        })

        metrics = diag.analyze_baseline(
            events=unstable_events,
            baseline_range=baseline_range,
            isotope_windows=isotope_windows,
            calibration=calibration
        )

        # Unstable data should be flagged
        assert metrics.is_stable is False
        assert len(metrics.warnings) > 0


class TestBuildBaselineDiagnostics:
    """Test the build_baseline_diagnostics function."""

    def test_build_baseline_diagnostics(self, sample_events, calibration, isotope_windows):
        """Test building baseline diagnostics."""
        baseline_info = {
            'start': sample_events['timestamp'].min(),
            'end': sample_events['timestamp'].min() + pd.Timedelta(hours=24)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_baseline_diagnostics(
                events=sample_events,
                baseline_info=baseline_info,
                calibration=calibration,
                isotope_windows=isotope_windows,
                output_dir=tmpdir
            )

            assert isinstance(result, dict)
            assert 'mean_rate' in result
            assert 'stability_score' in result
            assert 'is_stable' in result
            assert 'diagnostic_plot' in result
            assert 'diagnostic_report' in result

            # Check that files were created
            assert os.path.exists(result['diagnostic_plot'])
            assert os.path.exists(result['diagnostic_report'])

    def test_build_baseline_diagnostics_no_baseline_range(self, sample_events, calibration):
        """Test with missing baseline range."""
        baseline_info = {}  # No start/end

        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_baseline_diagnostics(
                events=sample_events,
                baseline_info=baseline_info,
                calibration=calibration,
                output_dir=tmpdir
            )

            # Should return empty dict
            assert result == {}

    def test_build_baseline_diagnostics_default_windows(self, sample_events, calibration):
        """Test with default isotope windows."""
        baseline_info = {
            'start': sample_events['timestamp'].min(),
            'end': sample_events['timestamp'].min() + pd.Timedelta(hours=24)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_baseline_diagnostics(
                events=sample_events,
                baseline_info=baseline_info,
                calibration=calibration,
                isotope_windows=None,  # Use defaults
                output_dir=tmpdir
            )

            assert isinstance(result, dict)
            assert 'mean_rate' in result

    def test_build_baseline_diagnostics_error_handling(self, sample_events, calibration):
        """Test error handling in build_baseline_diagnostics."""
        baseline_info = {
            'start': sample_events['timestamp'].min(),
            'end': sample_events['timestamp'].min() + pd.Timedelta(hours=24)
        }

        # Use bad calibration to trigger error
        bad_calibration = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_baseline_diagnostics(
                events=sample_events,
                baseline_info=baseline_info,
                calibration=bad_calibration,
                output_dir=tmpdir
            )

            # Should return error dict
            assert 'error' in result


class TestIntegrationWithReporting:
    """Test integration with reporting module."""

    def test_reporting_integration(self):
        """Test that reporting module can import and use baseline diagnostics."""
        from reporting import DEFAULT_DIAGNOSTICS

        # Check that baseline_quality was added to DEFAULT_DIAGNOSTICS
        assert 'baseline_quality' in DEFAULT_DIAGNOSTICS
        assert DEFAULT_DIAGNOSTICS['baseline_quality'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
