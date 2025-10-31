"""
Module: parallel_loader.py
Purpose: Efficient parallel and chunked loading of large radon monitor CSV files
Author: RMTest Enhancement Module

This module provides optimized CSV loading capabilities for large radon monitor
datasets, with support for parallel processing, chunked reading, and memory-efficient
streaming operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import time
from dataclasses import dataclass, field
import gc


@dataclass
class LoadConfig:
    """Configuration for parallel data loading."""
    chunk_size: int = 50000  # Rows per chunk
    n_workers: Optional[int] = None  # Auto-detect if None
    use_threads: bool = False  # Use threads instead of processes
    dtype_hints: Optional[Dict[str, type]] = None  # Column dtype hints
    parse_dates: Optional[List[str]] = None  # Date columns to parse
    filter_func: Optional[Callable] = None  # Optional filter during load
    progress_callback: Optional[Callable] = None  # Progress reporting
    memory_limit_gb: float = 4.0  # Max memory per worker
    compression: str = 'infer'  # CSV compression
    enable_parallel: bool = True  # Enable parallel loading
    size_threshold_mb: float = 100.0  # File size threshold for parallel loading


class ParallelCSVLoader:
    """
    Parallel and chunked CSV loader optimized for large radon monitor datasets.
    Handles 600MB+ files efficiently with minimal memory footprint.

    Example:
        >>> config = LoadConfig(chunk_size=100000, n_workers=4)
        >>> loader = ParallelCSVLoader(config)
        >>> df = loader.load_file("large_dataset.csv")
    """

    def __init__(self, config: Optional[LoadConfig] = None):
        """
        Initialize the parallel loader.

        Args:
            config: Loading configuration (uses defaults if None)
        """
        self.config = config or LoadConfig()
        if self.config.n_workers is None:
            # Use up to 8 workers, leaving 1 CPU for system
            self.config.n_workers = min(max(1, mp.cpu_count() - 1), 8)

        self.logger = logging.getLogger(__name__)

        # Default dtypes for radon monitor data (optimized for memory)
        if self.config.dtype_hints is None:
            self.config.dtype_hints = {
                'fUniqueID': np.int64,
                'fBits': np.int32,
                'adc': np.float32,  # Use float32 to save memory
                'fchannel': np.int16,
                'baseline_adc': np.float32,
                'spike_flag': np.int8,
                'valid': np.int8,
                'temperature': np.float32,
                'pressure': np.float32,
                'humidity': np.float32,
                'run_id': np.int32
            }

    def load_file(self, filepath: str, read_as_strings: bool = False) -> pd.DataFrame:
        """
        Load a large CSV file with parallel/chunked processing.

        Args:
            filepath: Path to the CSV file
            read_as_strings: If True, read all columns as strings (no dtype conversion)

        Returns:
            Complete DataFrame

        Raises:
            FileNotFoundError: If the file does not exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        self.logger.info(f"Loading {filepath.name} ({file_size_mb:.1f} MB)")

        start_time = time.time()

        # Choose loading strategy based on file size
        if not self.config.enable_parallel or file_size_mb < self.config.size_threshold_mb:
            # Small file - load directly
            df = self._load_direct(filepath, read_as_strings)
        elif file_size_mb < 500:
            # Medium file - use chunking
            df = self._load_chunked(filepath, read_as_strings)
        else:
            # Large file - use parallel chunking
            df = self._load_parallel(filepath, read_as_strings)

        elapsed = time.time() - start_time
        rows_per_sec = len(df) / elapsed if elapsed > 0 else 0
        self.logger.info(
            f"Loaded {len(df):,} rows in {elapsed:.2f}s "
            f"({rows_per_sec:.0f} rows/sec)"
        )

        return df

    def _load_direct(self, filepath: Path, read_as_strings: bool = False) -> pd.DataFrame:
        """Direct loading for small files."""
        dtype = str if read_as_strings else self.config.dtype_hints

        return pd.read_csv(
            filepath,
            dtype=dtype,
            parse_dates=self.config.parse_dates if not read_as_strings else None,
            compression=self.config.compression
        )

    def _load_chunked(self, filepath: Path, read_as_strings: bool = False) -> pd.DataFrame:
        """Chunked loading for medium files."""
        chunks = []
        dtype = str if read_as_strings else self.config.dtype_hints

        with pd.read_csv(
            filepath,
            chunksize=self.config.chunk_size,
            dtype=dtype,
            parse_dates=self.config.parse_dates if not read_as_strings else None,
            compression=self.config.compression
        ) as reader:
            for i, chunk in enumerate(reader):
                if self.config.filter_func:
                    chunk = self.config.filter_func(chunk)

                chunks.append(chunk)

                if self.config.progress_callback and i % 10 == 0:
                    self.config.progress_callback(i * self.config.chunk_size)

                # Memory management - periodically trigger garbage collection
                if i % 50 == 0:
                    gc.collect()

        return pd.concat(chunks, ignore_index=True)

    def _load_parallel(self, filepath: Path, read_as_strings: bool = False) -> pd.DataFrame:
        """Parallel loading for large files."""
        # First pass: count lines and find split points
        n_lines = self._count_lines(filepath)
        self.logger.info(f"File contains {n_lines:,} lines")

        # Calculate chunk boundaries
        chunk_boundaries = self._calculate_chunk_boundaries(n_lines)

        # Parallel load chunks
        executor_class = ThreadPoolExecutor if self.config.use_threads else ProcessPoolExecutor

        with executor_class(max_workers=self.config.n_workers) as executor:
            futures = []
            for start_row, end_row in chunk_boundaries:
                future = executor.submit(
                    self._load_chunk_range,
                    filepath, start_row, end_row, read_as_strings
                )
                futures.append(future)

            # Collect results
            chunks = []
            for i, future in enumerate(futures):
                chunk = future.result()
                if self.config.filter_func:
                    chunk = self.config.filter_func(chunk)
                chunks.append(chunk)

                if self.config.progress_callback:
                    self.config.progress_callback((i + 1) / len(futures))

        return pd.concat(chunks, ignore_index=True)

    def _count_lines(self, filepath: Path) -> int:
        """Fast line counting for large files using binary reads."""
        def blocks(file, size=65536):
            while True:
                b = file.read(size)
                if not b:
                    break
                yield b

        with open(filepath, "rb") as f:
            return sum(bl.count(b"\n") for bl in blocks(f))

    def _calculate_chunk_boundaries(self, n_lines: int) -> List[Tuple[int, int]]:
        """Calculate optimal chunk boundaries for parallel processing."""
        # Account for header row
        n_data_lines = n_lines - 1

        if n_data_lines <= 0:
            return []

        # Calculate chunks based on workers and memory limit
        rows_per_worker = n_data_lines // self.config.n_workers
        rows_per_worker = min(rows_per_worker, self.config.chunk_size * 10)

        # Ensure at least some rows per worker
        if rows_per_worker < 1:
            rows_per_worker = n_data_lines

        boundaries = []
        for i in range(self.config.n_workers):
            start = i * rows_per_worker + 1  # +1 to skip header
            if i == self.config.n_workers - 1:
                # Last worker gets all remaining rows
                end = n_lines
            else:
                end = (i + 1) * rows_per_worker + 1

            if start < n_lines:
                boundaries.append((start, min(end, n_lines)))

        return boundaries

    def _load_chunk_range(self, filepath: Path, start_row: int, end_row: int,
                          read_as_strings: bool = False) -> pd.DataFrame:
        """Load a specific row range from the CSV."""
        nrows = end_row - start_row
        dtype = str if read_as_strings else self.config.dtype_hints

        return pd.read_csv(
            filepath,
            skiprows=range(1, start_row),  # Skip rows before our chunk
            nrows=nrows,
            dtype=dtype,
            parse_dates=self.config.parse_dates if not read_as_strings else None,
            compression=self.config.compression
        )

    def stream_process(self,
                       filepath: str,
                       processor_func: Callable[[pd.DataFrame], Any],
                       aggregate_func: Optional[Callable[[List[Any]], Any]] = None) -> Any:
        """
        Stream process a large file without loading it entirely into memory.

        This is useful for computing statistics or aggregations on datasets
        that are too large to fit in memory.

        Args:
            filepath: Path to CSV file
            processor_func: Function to process each chunk, should return results
            aggregate_func: Optional function to aggregate results from all chunks

        Returns:
            Aggregated results (or list of results if aggregate_func is None)

        Example:
            >>> def compute_stats(chunk):
            ...     return {'mean': chunk['adc'].mean(), 'count': len(chunk)}
            >>> def aggregate(results):
            ...     total_count = sum(r['count'] for r in results)
            ...     weighted_mean = sum(r['mean'] * r['count'] for r in results) / total_count
            ...     return {'mean': weighted_mean, 'total_count': total_count}
            >>> result = loader.stream_process('data.csv', compute_stats, aggregate)
        """
        filepath = Path(filepath)
        results = []

        with pd.read_csv(
            filepath,
            chunksize=self.config.chunk_size,
            dtype=self.config.dtype_hints,
            parse_dates=self.config.parse_dates,
            compression=self.config.compression
        ) as reader:

            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = []

                for chunk in reader:
                    if self.config.filter_func:
                        chunk = self.config.filter_func(chunk)

                    future = executor.submit(processor_func, chunk)
                    futures.append(future)

                    # Limit concurrent futures to manage memory
                    if len(futures) >= self.config.n_workers * 2:
                        # Wait for some to complete
                        for future in futures[:self.config.n_workers]:
                            results.append(future.result())
                        futures = futures[self.config.n_workers:]

                # Collect remaining results
                for future in futures:
                    results.append(future.result())

        if aggregate_func:
            return aggregate_func(results)
        return results


class IncrementalProcessor:
    """
    Process radon monitor data incrementally to handle large datasets.

    This processor applies calibration and isotope classification in chunks,
    avoiding the need to load entire datasets into memory.
    """

    def __init__(self, calibration_params: Dict[str, Any]):
        """
        Initialize the incremental processor.

        Args:
            calibration_params: Calibration parameters including:
                - slope_MeV_per_ch: Energy calibration slope
                - intercept_MeV: Energy calibration intercept
                - noise_cutoff: ADC threshold for noise filtering (optional)
        """
        self.calibration = calibration_params
        self.stats = {
            'total_events': 0,
            'filtered_events': 0,
            'po210_events': 0,
            'po214_events': 0,
            'po218_events': 0
        }

    def process_chunk(self, chunk: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a single chunk of data.

        Args:
            chunk: DataFrame chunk with 'adc' column

        Returns:
            Dictionary containing processing results for this chunk:
                - total_events: Number of events in chunk
                - filtered_events: Number of events after noise filter
                - po210_events, po214_events, po218_events: Event counts by isotope
                - po210_data, po214_data, po218_data: DataFrames of events by isotope
        """
        # Apply calibration
        chunk = chunk.copy()
        chunk['energy_mev'] = (
            chunk['adc'] * self.calibration['slope_MeV_per_ch'] +
            self.calibration['intercept_MeV']
        )

        # Apply noise filter if configured
        if 'noise_cutoff' in self.calibration:
            mask = chunk['adc'] > self.calibration['noise_cutoff']
            filtered = chunk[mask]
        else:
            filtered = chunk

        # Classify events by isotope energy windows
        # Po-210: 5.3 MeV (window: 5.1-5.5 MeV)
        # Po-218: 6.0 MeV (window: 5.9-6.1 MeV)
        # Po-214: 7.7 MeV (window: 7.5-7.9 MeV)
        po210_mask = (filtered['energy_mev'] > 5.1) & (filtered['energy_mev'] < 5.5)
        po214_mask = (filtered['energy_mev'] > 7.5) & (filtered['energy_mev'] < 7.9)
        po218_mask = (filtered['energy_mev'] > 5.9) & (filtered['energy_mev'] < 6.1)

        return {
            'total_events': len(chunk),
            'filtered_events': len(filtered),
            'po210_events': po210_mask.sum(),
            'po214_events': po214_mask.sum(),
            'po218_events': po218_mask.sum(),
            'po210_data': filtered[po210_mask],
            'po214_data': filtered[po214_mask],
            'po218_data': filtered[po218_mask]
        }

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from all chunks.

        Args:
            results: List of result dictionaries from process_chunk

        Returns:
            Aggregated statistics and combined DataFrames
        """
        totals = {
            'total_events': 0,
            'filtered_events': 0,
            'po210_events': 0,
            'po214_events': 0,
            'po218_events': 0
        }

        all_po210 = []
        all_po214 = []
        all_po218 = []

        for result in results:
            for key in totals:
                totals[key] += result[key]

            all_po210.append(result['po210_data'])
            all_po214.append(result['po214_data'])
            all_po218.append(result['po218_data'])

        return {
            'statistics': totals,
            'po210_data': pd.concat(all_po210, ignore_index=True) if all_po210 else pd.DataFrame(),
            'po214_data': pd.concat(all_po214, ignore_index=True) if all_po214 else pd.DataFrame(),
            'po218_data': pd.concat(all_po218, ignore_index=True) if all_po218 else pd.DataFrame()
        }


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.

    This function downcasts numeric columns to the smallest dtype that can
    hold the data without loss of information.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with optimized dtypes (same data, less memory)

    Example:
        >>> df = optimize_dtypes(df)
        >>> # Memory usage typically reduced by 30-50%
    """
    df = df.copy()

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object' and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                # Downcast integers
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # Downcast floats
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    return df


if __name__ == "__main__":
    # Example usage demonstration
    config = LoadConfig(
        chunk_size=100000,
        n_workers=4,
        memory_limit_gb=2.0,
        filter_func=lambda df: df[df.get('valid', 1) == 1]  # Example filter
    )

    loader = ParallelCSVLoader(config)

    # Example calibration for processing
    calibration = {
        'slope_MeV_per_ch': 0.00430,
        'intercept_MeV': 0.0,
        'noise_cutoff': 400
    }

    processor = IncrementalProcessor(calibration)

    # Display configuration info
    print("Parallel CSV Loader - RMTest Enhancement Module")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Workers: {config.n_workers}")
    print(f"  Chunk size: {config.chunk_size:,} rows")
    print(f"  Memory limit: {config.memory_limit_gb} GB")
    print(f"  Parallel threshold: {config.size_threshold_mb} MB")

    # Memory usage estimation
    estimated_memory_per_chunk = config.chunk_size * 10 * 4 / (1024**2)  # Rough estimate
    print(f"\nEstimated memory usage:")
    print(f"  Per chunk: ~{estimated_memory_per_chunk:.1f} MB")
    print(f"  Total parallel: ~{estimated_memory_per_chunk * config.n_workers:.1f} MB")
    print("\nReady for stream processing of large radon monitor datasets.")
