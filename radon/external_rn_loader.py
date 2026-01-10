"""Helpers for loading external radon concentration time-series data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import logging

import pandas as pd

from utils.time_utils import parse_timestamp


__all__ = ["load_external_rn_series"]


logger = logging.getLogger(__name__)


def _get_constant(cfg_external: dict | None) -> float | None:
    """Get constant/fallback value from config, checking both old and new parameter names."""
    if not cfg_external:
        return 80.0

    # Check new parameter name first
    if "fallback_bq_per_m3" in cfg_external:
        return cfg_external.get("fallback_bq_per_m3")

    # Fall back to old parameter names for backwards compatibility
    if "constant_bq_per_m3" in cfg_external:
        return cfg_external.get("constant_bq_per_m3")

    if "default_bq_per_m3" in cfg_external:
        return cfg_external.get("default_bq_per_m3")

    return None


def _normalize_target_timestamps(target_timestamps: Iterable) -> list[pd.Timestamp]:
    return [parse_timestamp(ts) for ts in target_timestamps]


def _reindex_series(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    interpolation: str,
    allowed_skew_seconds: float,
) -> pd.Series:
    if series.empty:
        return pd.Series(index=target_index, dtype="float64")

    interpolation = interpolation.lower()
    series = series.sort_index()
    tolerance = pd.Timedelta(seconds=float(allowed_skew_seconds))
    if interpolation not in {"nearest", "ffill"}:
        raise ValueError(
            "external radon interpolation must be 'nearest' or 'ffill', "
            f"got {interpolation!r}"
        )

    if interpolation == "nearest":
        reindexed = series.reindex(target_index, method="nearest", tolerance=tolerance)
    else:
        reindexed = series.reindex(target_index, method="ffill", tolerance=tolerance)
    return reindexed


def _load_file_series(cfg_external: dict, target_index: pd.DatetimeIndex) -> pd.Series:
    file_path = cfg_external.get("file_path")
    if not file_path:
        raise ValueError("external radon config missing 'file_path'")

    # Validate file extension
    file_path_obj = Path(file_path).expanduser()
    if file_path_obj.suffix.lower() not in ['.csv', '.txt']:
        raise ValueError(
            f"external radon file must be CSV format, got {file_path_obj.suffix} for {file_path!r}"
        )

    try:
        df = pd.read_csv(file_path_obj)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "external radon file not found; check "
            "radon_inference.external_rn.file_path"
        ) from exc
    except OSError as exc:  # pragma: no cover - difficult to simulate precisely
        raise OSError(
            "external radon file not readable; check "
            "radon_inference.external_rn.file_path"
        ) from exc

    time_column = cfg_external.get("time_column", "timestamp")
    value_column = cfg_external.get("value_column", "rn_bq_per_m3")
    tz_name = cfg_external.get("timezone")

    if time_column not in df.columns:
        raise ValueError(
            f"external radon column {time_column!r} not found in file {file_path!r}"
        )
    if value_column not in df.columns:
        raise ValueError(
            f"external radon column {value_column!r} not found in file {file_path!r}"
        )

    timestamps = pd.to_datetime(df[time_column], utc=False, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("external radon time column contains unparsable timestamps")

    # Validate that value column is numeric before conversion
    values = pd.to_numeric(df[value_column], errors="coerce")
    if values.isna().any() and not df[value_column].isna().any():
        raise ValueError(
            f"external radon value column {value_column!r} contains non-numeric values"
        )
    series = pd.Series(values.to_numpy(dtype="float64"), index=timestamps)
    idx = pd.DatetimeIndex(series.index)
    if idx.tz is None:
        if tz_name:
            idx = idx.tz_localize(tz_name, nonexistent="shift_forward", ambiguous="NaT")
        else:
            idx = idx.tz_localize("UTC")
    else:
        if tz_name:
            logger.warning(
                "external radon time column already timezone aware; ignoring configured timezone"
            )
    idx = idx.tz_convert("UTC")
    series.index = idx
    series = series[~series.index.isna()]
    series = series[~series.index.duplicated(keep="last")]

    interpolation = cfg_external.get("interpolation", "nearest")
    # Check for new parameter name first, fall back to old name for backwards compatibility
    max_gap = cfg_external.get("max_gap_seconds")
    if max_gap is None:
        max_gap = cfg_external.get("allowed_skew_seconds", 300)
    return _reindex_series(series, target_index, interpolation, max_gap)


def load_external_rn_series(cfg_external: dict | None, target_timestamps):
    """Return ambient radon concentrations aligned with ``target_timestamps``."""

    target_ts = _normalize_target_timestamps(target_timestamps)
    if not target_ts:
        return []

    target_index = pd.DatetimeIndex(target_ts)
    constant_value = _get_constant(cfg_external)

    mode = (cfg_external or {}).get("mode", "constant").lower()

    if mode == "constant":
        if constant_value is None:
            raise ValueError("external radon constant mode requires a value")
        values = pd.Series(constant_value, index=target_index)
    elif mode == "file":
        try:
            values = _load_file_series(cfg_external or {}, target_index)
        except FileNotFoundError:
            if constant_value is None:
                raise
            logger.warning(
                "external radon file missing; falling back to constant value"
            )
            values = pd.Series(constant_value, index=target_index)
        except OSError:
            if constant_value is None:
                raise
            logger.warning(
                "external radon file unreadable; falling back to constant value"
            )
            values = pd.Series(constant_value, index=target_index)
        else:
            if values.isna().any():
                if constant_value is None:
                    missing = values[values.isna()].index[0]
                    max_gap = cfg_external.get("max_gap_seconds") or cfg_external.get("allowed_skew_seconds", 300)
                    raise ValueError(
                        f"No ambient radon value for timestamp {missing.isoformat()} "
                        f"(gap exceeds max_gap_seconds={max_gap}s) and no fallback defined. "
                        f"Add 'fallback_bq_per_m3' to config or increase 'max_gap_seconds'."
                    )
                logger.info(
                    "Filling %d missing external radon values with fallback: %.1f Bq/m³",
                    values.isna().sum(),
                    constant_value,
                )
                values = values.fillna(constant_value)
    else:
        raise ValueError(f"unsupported external radon mode: {mode!r}")

    return list(zip(target_ts, values.to_list()))

