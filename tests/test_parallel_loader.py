"""Tests for the parallel CSV loader module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from utils.parallel_loader import (
    ParallelCSVLoader,
    LoadConfig,
    IncrementalProcessor,
    optimize_dtypes,
)


@pytest.fixture
def sample_csv():
    """Create a temporary CSV file for testing."""
    data = {
        'fUniqueID': range(100),
        'fBits': [50331648] * 100,
        'timestamp': ['2023-01-01T00:00:00Z'] * 100,
        'adc': np.random.randint(400, 2000, 100),
        'fchannel': [0] * 100,
        'baseline_adc': [50] * 100,
        'spike_flag': [0] * 100,
        'valid': [1] * 100,
        'temperature': np.random.uniform(20, 25, 100),
        'run_id': [1000] * 100,
        'pressure': [1012] * 100,
        'humidity': [40] * 100,
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_load_config_defaults():
    """Test LoadConfig initialization with defaults."""
    config = LoadConfig()
    assert config.chunk_size == 50000
    assert config.compression == 'infer'
    assert config.enable_parallel is True


def test_load_config_custom():
    """Test LoadConfig with custom parameters."""
    config = LoadConfig(
        chunk_size=10000,
        n_workers=2,
        use_threads=True,
        size_threshold_mb=50.0
    )
    assert config.chunk_size == 10000
    assert config.n_workers == 2
    assert config.use_threads is True
    assert config.size_threshold_mb == 50.0


def test_parallel_loader_initialization():
    """Test ParallelCSVLoader initialization."""
    config = LoadConfig(n_workers=4)
    loader = ParallelCSVLoader(config)
    assert loader.config.n_workers == 4
    assert loader.config.dtype_hints is not None


def test_load_small_file(sample_csv):
    """Test loading a small CSV file."""
    config = LoadConfig(chunk_size=50, n_workers=2)
    loader = ParallelCSVLoader(config)

    df = loader.load_file(sample_csv, read_as_strings=True)

    assert len(df) == 100
    assert 'fUniqueID' in df.columns
    assert 'adc' in df.columns
    # All columns should be strings when read_as_strings=True
    assert df['adc'].dtype == object


def test_load_file_not_found():
    """Test loading a non-existent file raises FileNotFoundError."""
    loader = ParallelCSVLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_file('/nonexistent/file.csv')


def test_parallel_vs_direct_loading(sample_csv):
    """Test that parallel and direct loading produce same results."""
    config = LoadConfig(chunk_size=30, n_workers=2)
    loader = ParallelCSVLoader(config)

    # Load with direct method (force by disabling parallel)
    config_direct = LoadConfig(enable_parallel=False)
    loader_direct = ParallelCSVLoader(config_direct)
    df_direct = loader_direct.load_file(sample_csv, read_as_strings=True)

    # Load with parallel method (force by lowering threshold)
    config_parallel = LoadConfig(
        enable_parallel=True,
        size_threshold_mb=0.0,  # Force parallel even for tiny files
        chunk_size=30,
        n_workers=2
    )
    loader_parallel = ParallelCSVLoader(config_parallel)
    df_parallel = loader_parallel.load_file(sample_csv, read_as_strings=True)

    # Should have same number of rows
    assert len(df_direct) == len(df_parallel)
    # Should have same columns
    assert set(df_direct.columns) == set(df_parallel.columns)


def test_optimize_dtypes():
    """Test dtype optimization for memory reduction."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'large_int': [1000000, 2000000, 3000000, 4000000, 5000000],
    })

    optimized = optimize_dtypes(df)

    # Small integers should be downcast
    assert optimized['int_col'].dtype == np.int8
    # Floats should be downcast to float32
    assert optimized['float_col'].dtype == np.float32
    # Large ints should remain int32
    assert optimized['large_int'].dtype == np.int32


def test_incremental_processor():
    """Test incremental processing with calibration."""
    calibration = {
        'slope_MeV_per_ch': 0.00430,
        'intercept_MeV': 0.0,
        'noise_cutoff': 400
    }

    processor = IncrementalProcessor(calibration)

    # Create sample chunk
    chunk = pd.DataFrame({
        'adc': [500, 1200, 1800, 300, 1500],  # Mix of noise and signal
        'timestamp': pd.date_range('2023-01-01', periods=5)
    })

    result = processor.process_chunk(chunk)

    # Check result structure
    assert 'total_events' in result
    assert 'filtered_events' in result
    assert result['total_events'] == 5
    # Should filter out the 300 ADC event (below noise_cutoff)
    assert result['filtered_events'] == 4


def test_incremental_processor_aggregate():
    """Test aggregation of incremental processing results."""
    calibration = {
        'slope_MeV_per_ch': 0.00430,
        'intercept_MeV': 0.0,
    }

    processor = IncrementalProcessor(calibration)

    # Create two chunks
    chunk1 = pd.DataFrame({'adc': [1200, 1800], 'timestamp': pd.date_range('2023-01-01', periods=2)})
    chunk2 = pd.DataFrame({'adc': [1500, 1600], 'timestamp': pd.date_range('2023-01-03', periods=2)})

    result1 = processor.process_chunk(chunk1)
    result2 = processor.process_chunk(chunk2)

    aggregated = processor.aggregate_results([result1, result2])

    assert 'statistics' in aggregated
    assert aggregated['statistics']['total_events'] == 4
    assert aggregated['statistics']['filtered_events'] == 4


def test_real_test_data():
    """Test loading actual test data from the repository."""
    test_csv = Path('tests/data/mini_run/run.csv')

    if not test_csv.exists():
        pytest.skip("Test data not found")

    config = LoadConfig(chunk_size=10, n_workers=2)
    loader = ParallelCSVLoader(config)

    df = loader.load_file(test_csv, read_as_strings=True)

    # Check that we loaded the expected columns
    expected_cols = [
        'fUniqueID', 'fBits', 'timestamp', 'adc', 'fchannel',
        'baseline_adc', 'spike_flag', 'valid', 'temperature',
        'run_id', 'pressure', 'humidity'
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    assert len(df) > 0, "Should have loaded some rows"


def test_count_lines():
    """Test line counting functionality."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("header1,header2\n")
        for i in range(100):
            f.write(f"{i},value{i}\n")
        temp_path = f.name

    try:
        loader = ParallelCSVLoader()
        line_count = loader._count_lines(Path(temp_path))
        # Should count header + 100 data lines = 101
        assert line_count == 101
    finally:
        os.unlink(temp_path)


def test_calculate_chunk_boundaries():
    """Test chunk boundary calculation."""
    config = LoadConfig(n_workers=4, chunk_size=50000)
    loader = ParallelCSVLoader(config)

    # Test with 1000 lines (999 data lines + 1 header)
    boundaries = loader._calculate_chunk_boundaries(1000)

    # Should have chunks for each worker
    assert len(boundaries) > 0
    # First boundary should start at row 1 (skip header)
    assert boundaries[0][0] == 1
    # Last boundary should end at line 1000
    assert boundaries[-1][1] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
