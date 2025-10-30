"""Utilities for loading external ambient radon time-series data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import timedelta
import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


DEFAULT_INTERPOLATION = "nearest"
DEFAULT_ALLOWED_SKEW_SECONDS = 300
DEFAULT_BQ_PER_M3 = 80.0


def _to_utc_index(values: Iterable[Any], timezone: str | None = None) -> pd.DatetimeIndex:
    """Return ``values`` parsed to a UTC :class:`~pandas.DatetimeIndex`."""

    dt_index = pd.DatetimeIndex(pd.to_datetime(list(values)))

    if timezone:
        if dt_index.tz is None:
            dt_index = dt_index.tz_localize(timezone)
        else:
            dt_index = dt_index.tz_convert(timezone)
    elif dt_index.tz is None:
        dt_index = dt_index.tz_localize("UTC")

    return dt_index.tz_convert("UTC")


def _target_index(target_timestamps: Sequence[Any]) -> pd.DatetimeIndex:
    if not target_timestamps:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.DatetimeIndex(pd.to_datetime(target_timestamps, utc=True))


def _constant_value(cfg_external: Mapping[str, Any]) -> tuple[float | None, bool]:
    if cfg_external is None:
        return float(DEFAULT_BQ_PER_M3), True

    constant = cfg_external.get("constant_bq_per_m3")
    default = (
        cfg_external.get("default_bq_per_m3")
        if "default_bq_per_m3" in cfg_external
        else DEFAULT_BQ_PER_M3
    )

    if constant is not None:
        return float(constant), True
    if default is not None:
        return float(default), True
    return None, False


def load_external_rn_series(
    cfg_external: Mapping[str, Any] | None, target_timestamps: Sequence[Any]
) -> list[tuple[pd.Timestamp, float]]:
    """Return ambient radon data aligned to ``target_timestamps``.

    Parameters
    ----------
    cfg_external:
        Sub-configuration from ``config["radon_inference"]["external_rn"]``.
    target_timestamps:
        Iterable of timestamps (strings, ``datetime`` instances, or
        :class:`~pandas.Timestamp`).

    Returns
    -------
    list
        Tuples ``(timestamp, value)`` aligned with ``target_timestamps``.
    """

    constant_value, has_constant = _constant_value(cfg_external or {})

    index = _target_index(target_timestamps)
    if index.empty:
        return []

    mode = (cfg_external or {}).get("mode", "constant")

    if mode == "constant":
        if constant_value is None:
            raise ValueError("external radon constant_bq_per_m3 is not configured")
        values = [float(constant_value)] * len(index)
        return list(zip(list(index), values))

    if mode != "file":
        raise ValueError(f"unsupported external radon mode: {mode!r}")

    cfg = cfg_external or {}
    file_path = cfg.get("file_path")
    if not file_path:
        if constant_value is not None:
            logger.warning(
                "external radon configuration missing file_path; falling back to constant"
            )
            values = [float(constant_value)] * len(index)
            return list(zip(list(index), values))
        raise ValueError("external radon file_path is required when mode='file'")

    path = Path(file_path)

    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, OSError) as exc:
        if not has_constant:
            raise FileNotFoundError(f"external radon file not found: {path}") from exc
        logger.warning(
            "Could not read external radon file '%s'; falling back to constant", path
        )
        values = [float(constant_value)] * len(index)
        return list(zip(list(index), values))
    except Exception as exc:  # pragma: no cover - unexpected read errors
        if not has_constant:
            raise RuntimeError(
                f"failed to read external radon file {path}: {exc}"
            ) from exc
        logger.warning(
            "Failed to load external radon file '%s' (%s); falling back to constant",
            path,
            exc,
        )
        values = [float(constant_value)] * len(index)
        return list(zip(list(index), values))

    time_column = cfg.get("time_column", "timestamp")
    value_column = cfg.get("value_column", "rn_bq_per_m3")

    if time_column not in df.columns:
        raise KeyError(f"external radon time column '{time_column}' not found in file")
    if value_column not in df.columns:
        raise KeyError(
            f"external radon value column '{value_column}' not found in file"
        )

    series_index = _to_utc_index(df[time_column], cfg.get("timezone"))
    series = pd.Series(df[value_column].astype(float).to_numpy(), index=series_index)
    series = series[~series.index.duplicated(keep="last")].sort_index()

    if series.empty:
        if constant_value is None:
            raise ValueError(
                "external radon file contains no data and no constant fallback is provided"
            )
        values = [float(constant_value)] * len(index)
        return list(zip(list(index), values))

    interpolation = cfg.get("interpolation", DEFAULT_INTERPOLATION)
    allowed_skew = cfg.get(
        "allowed_skew_seconds", DEFAULT_ALLOWED_SKEW_SECONDS
    )
    tolerance = (
        timedelta(seconds=float(allowed_skew))
        if allowed_skew is not None
        else None
    )

    reindex_kwargs: dict[str, Any] = {}
    if tolerance is not None:
        reindex_kwargs["tolerance"] = tolerance

    if interpolation not in {"nearest", "ffill"}:
        raise ValueError(
            "external radon interpolation must be 'nearest' or 'ffill'"
        )

    try:
        aligned = series.reindex(index, method=interpolation, **reindex_kwargs)
    except ValueError as exc:
        raise ValueError(f"failed to align external radon series: {exc}") from exc

    if constant_value is not None:
        aligned = aligned.fillna(float(constant_value))

    if aligned.isna().any():
        missing_ts = aligned[aligned.isna()].index[0]
        raise ValueError(
            "no external radon value available for "
            f"{missing_ts.isoformat()} and no constant fallback configured"
        )

    return list(zip(list(aligned.index), aligned.astype(float).tolist()))


__all__ = ["load_external_rn_series"]
