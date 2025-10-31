"""Convert isotope count time series into inferred radon activity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_ISOTOPES = {"Po214", "Po218", "Po210"}


@dataclass
class _BinContribution:
    """Container holding the counts contributed by each isotope."""

    t: float
    dt: float
    counts: dict[str, float]


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
) -> list[_BinContribution]:
    bins: dict[float, _BinContribution] = {}

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

            key = round(t, 6)
            bin_entry = bins.get(key)
            if bin_entry is None:
                bin_entry = _BinContribution(t=t, dt=dt, counts={})
                bins[key] = bin_entry
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

    ordered = sorted(bins.values(), key=lambda item: item.t)
    return ordered


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
        Run configuration containing ``radon_inference`` options.
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

    bins = _prepare_bins(isotope_series, available_isotopes)
    if not bins:
        logger.info("Radon inference did not find any populated time bins; skipping")
        return None

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
        weighted_sum = 0.0
        contributing_isotopes = []
        for iso in available_isotopes:
            counts = bin_entry.counts.get(iso)
            if counts is None:
                continue
            eff = float(detection_map.get(iso, 0.0))
            if eff <= 0:
                continue
            activity = counts / (eff * dt)
            if chain_correction == "equilibrium":  # placeholder for future options
                activity = activity  # no-op but explicit for readability
            iso_activity[iso] = activity
            weighted_sum += weights.get(iso, 0.0) * activity
            contributing_isotopes.append(iso)

        if not contributing_isotopes:
            continue

        radon_bq = weighted_sum / overall_eff
        rn_entry = {
            "t": bin_entry.t,
            "dt": dt,
            "rn_bq": radon_bq,
            "source": contributing_isotopes[0]
            if len(contributing_isotopes) == 1
            else "weighted",
            "meta": {"contributions": iso_activity},
        }
        rn_series.append(rn_entry)

        ambient_val = ambient_lookup.get(round(bin_entry.t, 6))
        if ambient_val is not None:
            ambient_series_out.append(
                {"t": bin_entry.t, "rn_bq_per_m3": ambient_val, "dt": dt}
            )
            if ambient_val > 0:
                volume_m3 = radon_bq * dt / ambient_val
                dt_minutes = dt / 60.0 if dt > 0 else np.nan
                if dt_minutes > 0:
                    volume_lpm = volume_m3 * 1000.0 / dt_minutes
                else:
                    volume_lpm = float("nan")
                cumulative_volume += volume_m3
                volume_series.append(
                    {
                        "t": bin_entry.t,
                        "dt": dt,
                        "v_m3": volume_m3,
                        "v_lpm": volume_lpm,
                        "meta": {"ambient_rn_bq_m3": ambient_val},
                    }
                )
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

