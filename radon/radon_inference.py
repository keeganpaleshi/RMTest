"""Convert isotope count time series into inferred radon activity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_ISOTOPES = {"Po214", "Po218", "Po210"}


class InferenceMethod(Enum):
    """Radon inference method selection."""
    PO214 = "po214"
    PO218 = "po218"
    AVERAGE = "average"
    WEIGHTED = "weighted"  # Uses configured weights
    BEST = "best"  # Auto-select based on available data


@dataclass
class _BinContribution:
    """Container holding the counts contributed by each isotope.

    Attributes
    ----------
    t : float
        Timestamp in seconds since epoch
    dt : float
        Bin width in seconds
    counts : dict[str, float]
        Counts for each isotope
    uncertainties : dict[str, float]
        Poisson uncertainties for each isotope (optional)
    metadata : dict[str, object]
        Additional metadata about this bin (e.g., duplicates merged)
    """

    t: float
    dt: float
    counts: dict[str, float]
    uncertainties: dict[str, float] | None = None
    metadata: dict[str, object] | None = None


def _normalise_weights(
    isotopes: Sequence[str], weights_cfg: Mapping[str, float] | None
) -> dict[str, float]:
    if not isotopes:
        return {}

    weights = {}
    if weights_cfg:
        for iso in isotopes:
            try:
                weight = float(weights_cfg.get(iso, 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            weights[iso] = max(weight, 0.0)

    if not weights or sum(weights.values()) <= 0:
        default_weight = 1.0 / float(len(isotopes))
        return {iso: default_weight for iso in isotopes}

    total = sum(weights.values())
    return {iso: weights.get(iso, 0.0) / total for iso in isotopes}


def _prepare_bins(
    isotope_series: Mapping[str, Sequence[Mapping[str, float]]],
    isotopes: Sequence[str],
    dedupe_strategy: str = "last",
    dedupe_time_tolerance_us: int = 1000,
) -> tuple[list[_BinContribution], int]:
    """Prepare time bins from isotope series with deduplication.

    Parameters
    ----------
    isotope_series : Mapping[str, Sequence[Mapping[str, float]]]
        Isotope time series data
    isotopes : Sequence[str]
        List of isotopes to process
    dedupe_strategy : str, optional
        Strategy for handling duplicates: 'first', 'last', or 'average' (default: 'last')
    dedupe_time_tolerance_us : int, optional
        Time tolerance in microseconds for considering entries as duplicates (default: 1000)

    Returns
    -------
    tuple[list[_BinContribution], int]
        Tuple of (ordered bins, number of duplicates removed)
    """
    # Collect all entries with isotope tracking for deduplication
    all_entries: dict[tuple[str, float], list[tuple[float, float, float]]] = {}
    duplicates_count = 0

    for iso in isotopes:
        entries = isotope_series.get(iso) or []
        for entry in entries:
            try:
                t = float(entry.get("t"))
                dt = float(entry.get("dt"))
                counts = float(entry.get("counts"))
            except (TypeError, ValueError, AttributeError):
                continue

            if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0:
                continue

            # Use microsecond precision for grouping duplicates
            key = (iso, round(t, 6))
            if key not in all_entries:
                all_entries[key] = []
            all_entries[key].append((t, dt, counts))

    # Process entries according to deduplication strategy
    bins: dict[float, _BinContribution] = {}

    for (iso, t_rounded), entry_list in all_entries.items():
        if len(entry_list) > 1:
            duplicates_count += len(entry_list) - 1
            logger.debug(
                "Found %d duplicate entries for %s at t=%.6f, using strategy '%s'",
                len(entry_list),
                iso,
                t_rounded,
                dedupe_strategy,
            )

        # Apply deduplication strategy
        if dedupe_strategy == "first":
            t, dt, counts = entry_list[0]
        elif dedupe_strategy == "average":
            t = np.mean([e[0] for e in entry_list])
            dt = np.mean([e[1] for e in entry_list])
            counts = np.mean([e[2] for e in entry_list])
        else:  # "last" is default
            t, dt, counts = entry_list[-1]

        # Get or create bin
        bin_key = round(t, 6)
        bin_entry = bins.get(bin_key)
        if bin_entry is None:
            bin_entry = _BinContribution(
                t=t,
                dt=dt,
                counts={},
                uncertainties={},
                metadata={"duplicates_removed": 0}
            )
            bins[bin_key] = bin_entry
        else:
            if not np.isclose(bin_entry.dt, dt):
                logger.debug(
                    "Bin width mismatch for isotope %s at t=%s: %s vs %s",
                    iso,
                    t,
                    bin_entry.dt,
                    dt,
                )

        bin_entry.counts[iso] = counts
        # Poisson uncertainty: sqrt(N)
        bin_entry.uncertainties[iso] = np.sqrt(max(counts, 1.0))

        # Track duplicates in metadata
        if len(entry_list) > 1:
            bin_entry.metadata["duplicates_removed"] += len(entry_list) - 1

    ordered = sorted(bins.values(), key=lambda item: item.t)
    return ordered, duplicates_count


def _interpolate_external(
    ambient_series: Iterable[Mapping[str, float]] | None,
    times: Sequence[float],
) -> list[tuple[float, float]]:
    if not ambient_series:
        return []

    values: list[tuple[float, float]] = []
    for entry in ambient_series:
        try:
            t = entry["t"] if isinstance(entry, Mapping) else entry[0]
            val = entry.get("rn_bq_per_m3") if isinstance(entry, Mapping) else entry[1]
            if hasattr(t, "timestamp"):
                t_val = float(t.timestamp())
            else:
                t_val = float(t)
            val_float = float(val)
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if not np.isfinite(t_val) or not np.isfinite(val_float):
            continue
        values.append((t_val, val_float))

    if not values:
        return []

    values.sort(key=lambda pair: pair[0])
    times_arr = np.asarray([t for t, _ in values], dtype=float)
    vals_arr = np.asarray([v for _, v in values], dtype=float)
    targets = np.asarray(times, dtype=float)

    if len(values) == 1:
        filled = np.full_like(targets, vals_arr[0], dtype=float)
    else:
        filled = np.interp(targets, times_arr, vals_arr, left=vals_arr[0], right=vals_arr[-1])

    return list(zip(targets.tolist(), filled.tolist()))


def run_radon_inference(
    isotope_series: Mapping[str, Sequence[Mapping[str, float]]],
    config: Mapping[str, object],
    external_rn_series: Iterable[Mapping[str, float]] | None = None,
):
    """Return inferred Rn-222 activity derived from isotope counts.

    Parameters
    ----------
    isotope_series
        Mapping of isotope names to sequences of entries containing ``t`` (seconds),
        ``counts`` and ``dt`` fields.
    config
        Run configuration containing ``radon_inference`` options. New options:
        - dedupe_strategy: 'first', 'last', or 'average' (default: 'last')
        - dedupe_time_tolerance_us: Time tolerance in microseconds (default: 1000)
        - inference_method: 'po214', 'po218', 'average', 'weighted', or 'best' (default: 'weighted')
        - track_uncertainties: Enable Poisson uncertainty tracking (default: True)
    external_rn_series
        Optional iterable of mappings providing ``t`` (seconds) and
        ``rn_bq_per_m3`` ambient radon values.
    """

    radon_cfg = config.get("radon_inference") if isinstance(config, Mapping) else None
    if not isinstance(radon_cfg, Mapping) or not radon_cfg.get("enabled", False):
        return None

    requested_isotopes = radon_cfg.get("source_isotopes") or []
    if not requested_isotopes:
        logger.debug("Radon inference requested without source isotopes; skipping stage")
        return None

    detection_map = radon_cfg.get("detection_efficiency") or {}
    if not isinstance(detection_map, Mapping):
        logger.warning("Invalid detection_efficiency configuration; skipping radon inference")
        return None

    # Get enhanced configuration options (with backward-compatible defaults)
    dedupe_strategy = str(radon_cfg.get("dedupe_strategy", "last")).lower()
    if dedupe_strategy not in {"first", "last", "average"}:
        logger.warning("Invalid dedupe_strategy '%s', defaulting to 'last'", dedupe_strategy)
        dedupe_strategy = "last"

    dedupe_time_tolerance_us = int(radon_cfg.get("dedupe_time_tolerance_us", 1000))
    track_uncertainties = bool(radon_cfg.get("track_uncertainties", True))

    # Get inference method
    inference_method_str = str(radon_cfg.get("inference_method", "weighted")).lower()
    try:
        inference_method = InferenceMethod(inference_method_str)
    except ValueError:
        logger.warning("Invalid inference_method '%s', defaulting to 'weighted'", inference_method_str)
        inference_method = InferenceMethod.WEIGHTED

    available_isotopes: list[str] = []
    for iso in requested_isotopes:
        if iso not in SUPPORTED_ISOTOPES:
            logger.warning("Unsupported isotope %s in radon inference configuration", iso)
            continue
        if not isotope_series.get(iso):
            logger.warning("Radon inference missing time series for %s; skipping", iso)
            continue
        eff_val = detection_map.get(iso)
        try:
            eff = float(eff_val)
        except (TypeError, ValueError):
            eff = 0.0
        if eff <= 0:
            logger.warning(
                "Radon inference detection efficiency for %s is non-positive; skipping",
                iso,
            )
            continue
        available_isotopes.append(iso)

    if not available_isotopes:
        logger.info("Radon inference found no usable isotopes; stage skipped")
        return None

    # Auto-select method if using 'best'
    if inference_method == InferenceMethod.BEST:
        if "Po214" in available_isotopes and "Po218" in available_isotopes:
            inference_method = InferenceMethod.WEIGHTED
        elif "Po214" in available_isotopes:
            inference_method = InferenceMethod.PO214
        elif "Po218" in available_isotopes:
            inference_method = InferenceMethod.PO218
        else:
            inference_method = InferenceMethod.WEIGHTED
        logger.debug("Auto-selected inference method: %s", inference_method.value)

    weights = _normalise_weights(available_isotopes, radon_cfg.get("source_weights"))
    transport_eff = float(radon_cfg.get("transport_efficiency", 1.0) or 1.0)
    retention_eff = float(radon_cfg.get("retention_efficiency", 1.0) or 1.0)
    overall_eff = transport_eff * retention_eff
    if overall_eff <= 0:
        logger.warning(
            "Transport/retention efficiency product is non-positive; defaulting to 1.0",
        )
        overall_eff = 1.0

    chain_correction = str(radon_cfg.get("chain_correction", "none")).lower()
    if chain_correction not in {"none", "equilibrium"}:
        logger.warning("Unknown chain_correction %s – defaulting to 'none'", chain_correction)
        chain_correction = "none"

    bins, duplicates_removed = _prepare_bins(
        isotope_series,
        available_isotopes,
        dedupe_strategy=dedupe_strategy,
        dedupe_time_tolerance_us=dedupe_time_tolerance_us
    )
    if not bins:
        logger.info("Radon inference did not find any populated time bins; skipping")
        return None

    if duplicates_removed > 0:
        logger.info(
            "Radon inference removed %d duplicate entries using '%s' strategy",
            duplicates_removed,
            dedupe_strategy
        )

    times = [bin_entry.t for bin_entry in bins]
    ambient_aligned = _interpolate_external(external_rn_series, times)
    ambient_lookup = {round(t, 6): val for t, val in ambient_aligned}

    rn_series: list[dict[str, object]] = []
    volume_series: list[dict[str, object]] = []
    ambient_series_out: list[dict[str, object]] = []
    cumulative_series: list[dict[str, object]] = []
    cumulative_volume = 0.0

    for bin_entry in bins:
        dt = bin_entry.dt
        iso_activity = {}
        iso_uncertainty = {}
        weighted_sum = 0.0
        weighted_variance = 0.0
        contributing_isotopes = []

        for iso in available_isotopes:
            counts = bin_entry.counts.get(iso)
            if counts is None:
                continue
            eff = float(detection_map.get(iso, 0.0))
            if eff <= 0:
                continue

            # Calculate activity: A = N / (ε * Δt)
            activity = counts / (eff * dt)
            if chain_correction == "equilibrium":  # placeholder for future options
                activity = activity  # no-op but explicit for readability

            # Calculate uncertainty from Poisson statistics: σ_A = sqrt(N) / (ε * Δt)
            if track_uncertainties and bin_entry.uncertainties:
                count_unc = bin_entry.uncertainties.get(iso, np.sqrt(max(counts, 1.0)))
                activity_unc = count_unc / (eff * dt)
            else:
                activity_unc = np.sqrt(max(counts, 1.0)) / (eff * dt)

            iso_activity[iso] = activity
            iso_uncertainty[iso] = activity_unc
            contributing_isotopes.append(iso)

        if not contributing_isotopes:
            continue

        # Apply inference method
        if inference_method == InferenceMethod.PO214 and "Po214" in iso_activity:
            radon_bq = iso_activity["Po214"] / overall_eff
            radon_unc = iso_uncertainty["Po214"] / overall_eff
            source = "Po214"
        elif inference_method == InferenceMethod.PO218 and "Po218" in iso_activity:
            radon_bq = iso_activity["Po218"] / overall_eff
            radon_unc = iso_uncertainty["Po218"] / overall_eff
            source = "Po218"
        elif inference_method == InferenceMethod.AVERAGE:
            # Simple average of available isotopes
            radon_bq = np.mean(list(iso_activity.values())) / overall_eff
            # Propagate uncertainty assuming independent measurements
            radon_unc = np.sqrt(np.sum([u**2 for u in iso_uncertainty.values()])) / (overall_eff * len(iso_uncertainty))
            source = "average"
        else:  # WEIGHTED or fallback
            # Weighted combination
            for iso in contributing_isotopes:
                w = weights.get(iso, 0.0)
                weighted_sum += w * iso_activity[iso]
                weighted_variance += (w * iso_uncertainty[iso]) ** 2

            radon_bq = weighted_sum / overall_eff
            radon_unc = np.sqrt(weighted_variance) / overall_eff
            source = contributing_isotopes[0] if len(contributing_isotopes) == 1 else "weighted"

        rn_entry = {
            "t": bin_entry.t,
            "dt": dt,
            "rn_bq": radon_bq,
            "source": source,
            "meta": {
                "contributions": iso_activity,
                "uncertainties": iso_uncertainty if track_uncertainties else {},
            },
        }

        # Add uncertainty to output if tracking
        if track_uncertainties:
            rn_entry["rn_bq_unc"] = radon_unc

        rn_series.append(rn_entry)

        ambient_val = ambient_lookup.get(round(bin_entry.t, 6))
        if ambient_val is not None:
            ambient_series_out.append(
                {"t": bin_entry.t, "rn_bq_per_m3": ambient_val, "dt": dt}
            )
            if ambient_val > 0:
                # Calculate equivalent volume: V = A * Δt / C
                # where A is activity (Bq), C is ambient concentration (Bq/m³)
                volume_m3 = radon_bq * dt / ambient_val
                dt_minutes = dt / 60.0 if dt > 0 else np.nan
                if dt_minutes > 0:
                    volume_lpm = volume_m3 * 1000.0 / dt_minutes
                else:
                    volume_lpm = float("nan")

                # Propagate uncertainty if available
                # σ_V = V * σ_A / A (assuming ambient uncertainty is negligible)
                # For full propagation: σ_V = V * sqrt((σ_A/A)² + (σ_C/C)²)
                volume_entry = {
                    "t": bin_entry.t,
                    "dt": dt,
                    "v_m3": volume_m3,
                    "v_lpm": volume_lpm,
                    "meta": {"ambient_rn_bq_m3": ambient_val},
                }

                if track_uncertainties and radon_bq > 0:
                    # Simple propagation (ambient uncertainty assumed small)
                    volume_unc_m3 = volume_m3 * (radon_unc / radon_bq)
                    volume_entry["v_m3_unc"] = volume_unc_m3
                    if dt_minutes > 0:
                        volume_entry["v_lpm_unc"] = volume_unc_m3 * 1000.0 / dt_minutes

                cumulative_volume += volume_m3
                volume_series.append(volume_entry)
                cumulative_series.append(
                    {"t": bin_entry.t, "v_m3_cum": cumulative_volume}
                )
        else:
            ambient_series_out.append({"t": bin_entry.t, "rn_bq_per_m3": None, "dt": dt})

    if not rn_series:
        logger.info("Radon inference did not compute any activity samples; skipping")
        return None

    meta = {
        "source_isotopes": available_isotopes,
        "source_weights": weights,
        "detection_efficiency": {
            iso: float(detection_map.get(iso)) for iso in available_isotopes
        },
        "transport_efficiency": transport_eff,
        "retention_efficiency": retention_eff,
        "chain_correction": chain_correction,
        "inference_method": inference_method.value,
        "dedupe_strategy": dedupe_strategy,
        "duplicates_removed": duplicates_removed,
        "track_uncertainties": track_uncertainties,
    }

    result = {
        "rn_inferred": rn_series,
        "ambient_rn": ambient_series_out,
        "volume_equiv": volume_series,
        "volume_cumulative": cumulative_series,
        "meta": meta,
    }
    return result


__all__ = ["run_radon_inference"]

