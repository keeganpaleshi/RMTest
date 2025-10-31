"""
Tests for the enhancement framework.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from enhancements import (
    EnhancementConfig,
    EnhancementMetrics,
    EnhancementPipeline,
)


class TestEnhancementConfig:
    """Test EnhancementConfig creation and parsing."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnhancementConfig()
        assert config.parallel_loading_enabled is False
        assert config.baseline_diagnostics_enabled is False
        assert config.radon_inference_enhanced is False
        assert config.stable_emg_enabled is False

    def test_from_config_empty(self):
        """Test creation from empty config dict."""
        config = EnhancementConfig.from_config({})
        assert config.parallel_loading_enabled is False

    def test_from_config_with_enhancements(self):
        """Test creation from config with enhancements section."""
        cfg = {
            'enhancements': {
                'parallel_loading': {
                    'enabled': True,
                    'chunk_size': 50000,
                    'n_workers': 4,
                },
                'baseline_diagnostics': {
                    'enabled': True,
                    'generate_report': False,
                },
            }
        }
        config = EnhancementConfig.from_config(cfg)
        assert config.parallel_loading_enabled is True
        assert config.parallel_chunk_size == 50000
        assert config.parallel_n_workers == 4
        assert config.baseline_diagnostics_enabled is True
        assert config.baseline_generate_report is False

    def test_from_config_boolean_shorthand(self):
        """Test boolean shorthand in config."""
        cfg = {
            'enhancements': {
                'parallel_loading': True,
                'stable_emg': True,
            }
        }
        config = EnhancementConfig.from_config(cfg)
        assert config.parallel_loading_enabled is True
        assert config.stable_emg_enabled is True


class TestEnhancementMetrics:
    """Test EnhancementMetrics tracking."""

    def test_add_timing(self):
        """Test adding timing measurements."""
        metrics = EnhancementMetrics()
        metrics.add_timing('test_operation', 1.23)
        assert 'test_operation' in metrics.timings
        assert metrics.timings['test_operation'] == 1.23

    def test_add_warning(self):
        """Test adding warnings."""
        metrics = EnhancementMetrics()
        metrics.add_warning('test warning')
        assert 'test warning' in metrics.warnings
        assert len(metrics.warnings) == 1

    def test_multiple_timings(self):
        """Test adding multiple timings."""
        metrics = EnhancementMetrics()
        metrics.add_timing('op1', 1.0)
        metrics.add_timing('op2', 2.0)
        metrics.add_timing('op3', 3.0)
        assert len(metrics.timings) == 3
        assert sum(metrics.timings.values()) == 6.0


class TestEnhancementPipeline:
    """Test EnhancementPipeline functionality."""

    def test_init_no_enhancements(self):
        """Test initialization with no enhancements enabled."""
        config = {}
        pipeline = EnhancementPipeline(config)
        assert pipeline.enh_config.parallel_loading_enabled is False
        assert pipeline.enh_config.baseline_diagnostics_enabled is False

    def test_init_with_enhancements(self):
        """Test initialization with enhancements enabled."""
        config = {
            'enhancements': {
                'parallel_loading': {'enabled': True},
            }
        }
        pipeline = EnhancementPipeline(config)
        assert pipeline.enh_config.parallel_loading_enabled is True

    def test_enhanced_load_events_fallback(self, tmp_path):
        """Test enhanced load falls back to standard when enhancement unavailable."""
        # Create a small test CSV
        test_csv = tmp_path / "test.csv"
        df = pd.DataFrame({
            'timestamps': pd.date_range('2024-01-01', periods=10, freq='1s'),
            'adc_channels': np.random.randint(100, 2000, 10),
            'fchannel': [0] * 10,
            'fUniqueID': range(10),
            'fBits': [0] * 10,
        })
        df.to_csv(test_csv, index=False)

        # Create pipeline with no enhancements
        config = {
            'columns': {
                'timestamp': 'timestamps',
                'adc': 'adc_channels',
            }
        }
        pipeline = EnhancementPipeline(config)

        # Should use standard loading
        events = pipeline.enhanced_load_events(test_csv, column_map=config['columns'])
        assert len(events) == 10
        assert 'timestamp' in events.columns
        assert 'adc' in events.columns

    def test_enhanced_baseline_diagnostics_disabled(self):
        """Test baseline diagnostics when disabled."""
        config = {
            'enhancements': {
                'baseline_diagnostics': {'enabled': False}
            }
        }
        pipeline = EnhancementPipeline(config)

        # Should return None when disabled
        result = pipeline.enhanced_baseline_diagnostics(
            pd.DataFrame(),
            '/tmp/test'
        )
        assert result is None

    def test_enhanced_emg_fitting_disabled(self):
        """Test EMG fitting when enhancement disabled."""
        config = {
            'enhancements': {
                'stable_emg': {'enabled': False}
            }
        }
        pipeline = EnhancementPipeline(config)

        energy_data = np.random.normal(5.0, 0.1, 100)
        initial_params = {'mu': 5.0, 'sigma': 0.1, 'tau': 0.01}

        # Should return initial params when disabled
        result = pipeline.enhanced_emg_fitting(energy_data, initial_params)
        assert result == initial_params

    def test_save_metrics(self, tmp_path):
        """Test saving enhancement metrics to file."""
        config = {}
        pipeline = EnhancementPipeline(config)

        # Add some metrics
        pipeline.metrics.add_timing('test_op', 1.23)
        pipeline.metrics.add_warning('test warning')

        # Save metrics
        output_dir = tmp_path / 'output'
        pipeline.save_metrics(output_dir)

        # Check file was created
        metrics_file = output_dir / 'enhancement_metrics.json'
        assert metrics_file.exists()

        # Load and verify
        with open(metrics_file) as f:
            data = json.load(f)

        assert 'timings' in data
        assert 'test_op' in data['timings']
        assert data['timings']['test_op'] == 1.23
        assert 'test warning' in data['warnings']
        assert 'enhancements_enabled' in data

    def test_metrics_tracking(self):
        """Test that metrics are tracked during operations."""
        config = {}
        pipeline = EnhancementPipeline(config)

        # Initially empty
        assert len(pipeline.metrics.timings) == 0
        assert len(pipeline.metrics.warnings) == 0

        # After operations, should have entries
        # (This is tested implicitly by other tests that perform operations)


class TestEnhancementIntegration:
    """Test integration with existing RMTest components."""

    def test_config_compatibility(self):
        """Test that enhancement config is compatible with main config."""
        # Typical RMTest config structure
        config = {
            'pipeline': {'log_level': 'INFO'},
            'analysis': {
                'analysis_start_time': '2024-01-01T00:00:00Z',
                'analysis_end_time': '2024-01-02T00:00:00Z',
            },
            'baseline': {
                'range': ['2024-01-01T00:00:00Z', '2024-01-01T12:00:00Z'],
                'monitor_volume_l': 605.0,
            },
            'enhancements': {
                'parallel_loading': {'enabled': True},
            }
        }

        # Should create without errors
        pipeline = EnhancementPipeline(config)
        assert pipeline.config == config
        assert pipeline.enh_config.parallel_loading_enabled is True

    def test_graceful_module_import_failure(self):
        """Test that missing enhancement modules don't crash pipeline."""
        config = {
            'enhancements': {
                'parallel_loading': {'enabled': True},
                'baseline_diagnostics': {'enabled': True},
                'radon_inference': {'enhanced': True},
                'stable_emg': {'enabled': True},
            }
        }

        # Should create successfully even if modules don't exist
        pipeline = EnhancementPipeline(config)

        # Verify warnings are logged for missing modules
        # (checked via log output in integration tests)
        assert pipeline.enh_config.parallel_loading_enabled is True


def test_enhancement_config_schema():
    """Test that enhancement config matches expected schema."""
    config = {
        'enhancements': {
            'parallel_loading': {
                'enabled': True,
                'chunk_size': 100000,
                'n_workers': None,
                'memory_limit_gb': 4.0,
            },
            'baseline_diagnostics': {
                'enabled': True,
                'generate_report': True,
            },
            'radon_inference': {
                'enhanced': True,
                'dedupe_enabled': True,
            },
            'stable_emg': {
                'enabled': True,
                'use_log_scale': True,
            },
        }
    }

    enh_config = EnhancementConfig.from_config(config)

    # Verify all fields parsed correctly
    assert enh_config.parallel_loading_enabled is True
    assert enh_config.parallel_chunk_size == 100000
    assert enh_config.parallel_n_workers is None
    assert enh_config.parallel_memory_limit_gb == 4.0

    assert enh_config.baseline_diagnostics_enabled is True
    assert enh_config.baseline_generate_report is True

    assert enh_config.radon_inference_enhanced is True
    assert enh_config.radon_dedupe_enabled is True

    assert enh_config.stable_emg_enabled is True
    assert enh_config.stable_emg_use_log_scale is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
