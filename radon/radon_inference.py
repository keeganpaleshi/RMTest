"""Convert isotope count time series into inferred radon activity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from constants import load_half_life_overrides

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




def _estimate_leak_rate(
    *,
    start_activity: float,
    mean_activity: float,
    ambient_activity: float,
    interval_dt: float,
    rn_decay_constant: float,
    initial_interval: bool = False,
    sigma_mean_activity: float = 0.0,
    sigma_start_activity: float = 0.0,
) -> dict[str, float | bool | str] | None:
    """Estimate leak rate from interval-averaged activity.

    Within one interval the radon inventory obeys ``dA/dt = q * C - lambda * A``.
    Treating ``mean_activity`` as the average inventory over that interval and
    ``start_activity`` as the inventory at the interval start gives a leak-rate
    estimate that is substantially less sensitive to binning than differencing
    adjacent averages directly.

    When *sigma_mean_activity* and/or *sigma_start_activity* are provided the
    returned dictionary also contains ``"leak_rate_err_m3_s"`` computed via
    first-order error propagation through the inventory-balance equation.
    """

    if (
        not np.isfinite(start_activity)
        or not np.isfinite(mean_activity)
        or not np.isfinite(ambient_activity)
        or ambient_activity <= 0.0
        or interval_dt <= 0.0
        or not np.isfinite(rn_decay_constant)
        or rn_decay_constant <= 0.0
    ):
        return None

    survival = float(np.exp(-rn_decay_constant * interval_dt))
    response = float(-np.expm1(-rn_decay_constant * interval_dt) / rn_decay_constant)
    mean_weight = response / interval_dt
    leak_weight = 1.0 - mean_weight
    if not np.isfinite(mean_weight) or not np.isfinite(leak_weight) or leak_weight <= 0.0:
        return None

    leak_rate_raw = (
        rn_decay_constant * (mean_activity - start_activity * mean_weight)
    ) / (ambient_activity * leak_weight)
    clipped_to_zero = bool(leak_rate_raw < 0.0)
    leak_rate_m3_s = max(float(leak_rate_raw), 0.0)
    end_activity = start_activity * survival + leak_rate_m3_s * ambient_activity * response
    method = "steady_state_initial" if initial_interval else "inventory_balance"
    if clipped_to_zero:
        method = f"{method}_clipped"

    # --- uncertainty propagation (first-order Jacobian) ---
    # L = lambda * (A_mean - A_start * w_mean) / (C_ambient * w_leak)
    denom = ambient_activity * leak_weight
    dL_dAmean = rn_decay_constant / denom
    dL_dAstart = -rn_decay_constant * mean_weight / denom
    sigma_leak = float(np.sqrt(
        (dL_dAmean * sigma_mean_activity) ** 2
        + (dL_dAstart * sigma_start_activity) ** 2
    ))
    # If leak rate was clipped to zero, the uncertainty is still meaningful
    # (it tells you how uncertain the "zero" is).

    return {
        "leak_rate_m3_s": leak_rate_m3_s,
        "leak_rate_err_m3_s": sigma_leak,
        "leak_rate_raw_m3_s": float(leak_rate_raw),
        "inventory_equiv_m3": float(mean_activity / ambient_activity),
        "mean_activity_bq": float(mean_activity),
        "start_activity_bq": float(start_activity),
        "end_activity_bq": float(end_activity),
        "clipped_to_zero": clipped_to_zero,
        "method": method,
    }


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
    if chain_correction not in {"none", "assume_equilibrium", "equilibrium", "forward_model"}:
        logger.warning("Unknown chain_correction %s --defaulting to 'none'", chain_correction)
        chain_correction = "none"

    bins = _prepare_bins(isotope_series, available_isotopes)
    if not bins:
        logger.info("Radon inference did not find any populated time bins; skipping")
        return None

    times = [bin_entry.t for bin_entry in bins]
    ambient_aligned = _interpolate_external(external_rn_series, times)
    ambient_lookup = {round(t, 6): val for t, val in ambient_aligned}

    half_life_map = load_half_life_overrides(dict(config) if isinstance(config, Mapping) else None)
    rn_half_life_s = float(half_life_map.get("Rn222", 0.0))
    if not np.isfinite(rn_half_life_s) or rn_half_life_s <= 0.0:
        logger.warning("Invalid Rn222 half-life configured for radon inference; skipping stage")
        return None
    rn_decay_constant = float(np.log(2.0) / rn_half_life_s)

    rn_series: list[dict[str, object]] = []
    volume_series: list[dict[str, object]] = []
    ambient_series_out: list[dict[str, object]] = []
    cumulative_series: list[dict[str, object]] = []
    cumulative_volume_m3 = 0.0
    negative_leak_rate_intervals_clipped = 0

    for bin_entry in bins:
        dt = bin_entry.dt
        iso_activity = {}
        weighted_sum = 0.0
        weighted_var = 0.0  # variance accumulator for uncertainty
        contributing_isotopes = []
        for iso in available_isotopes:
            counts = bin_entry.counts.get(iso)
            if counts is None:
                continue
            eff = float(detection_map.get(iso, 0.0))
            if eff <= 0:
                continue
            activity = counts / (eff * dt)
            # Poisson uncertainty: sigma_activity = sqrt(counts) / (eff * dt)
            sigma_activity = float(np.sqrt(max(counts, 0.0))) / (eff * dt)
            if chain_correction in ("equilibrium", "assume_equilibrium"):
                pass  # placeholder for future correction logic
            iso_activity[iso] = activity
            w = weights.get(iso, 0.0)
            weighted_sum += w * activity
            weighted_var += (w * sigma_activity) ** 2
            contributing_isotopes.append(iso)

        if not contributing_isotopes:
            continue

        radon_bq = weighted_sum / overall_eff
        radon_bq_err = float(np.sqrt(weighted_var)) / overall_eff
        current_t = float(bin_entry.t)
        rn_entry = {
            "t": current_t,
            "dt": dt,
            "rn_bq": radon_bq,
            "rn_bq_err": radon_bq_err,
            "source": contributing_isotopes[0]
            if len(contributing_isotopes) == 1
            else "weighted",
            "meta": {"contributions": iso_activity},
        }
        rn_series.append(rn_entry)

        ambient_val = ambient_lookup.get(round(current_t, 6))
        ambient_series_out.append(
            {"t": current_t, "rn_bq_per_m3": ambient_val, "dt": dt}
        )

    previous_end_activity: float | None = None
    previous_end_sigma: float = 0.0
    cumulative_volume_var = 0.0  # running variance for cumulative uncertainty
    for rn_entry in rn_series:
        current_t = float(rn_entry["t"])
        mean_activity = float(rn_entry["rn_bq"])
        sigma_mean = float(rn_entry.get("rn_bq_err", 0.0))
        interval_dt = float(rn_entry["dt"])
        current_ambient = ambient_lookup.get(round(current_t, 6))
        if current_ambient is None or not np.isfinite(current_ambient) or current_ambient <= 0.0:
            continue

        initial_interval = previous_end_activity is None
        start_activity = mean_activity if initial_interval else float(previous_end_activity)
        sigma_start = sigma_mean if initial_interval else previous_end_sigma
        leak_estimate = _estimate_leak_rate(
            start_activity=start_activity,
            mean_activity=mean_activity,
            ambient_activity=float(current_ambient),
            interval_dt=interval_dt,
            rn_decay_constant=rn_decay_constant,
            initial_interval=initial_interval,
            sigma_mean_activity=sigma_mean,
            sigma_start_activity=sigma_start,
        )
        if leak_estimate is None:
            continue

        if leak_estimate["clipped_to_zero"]:
            negative_leak_rate_intervals_clipped += 1

        leak_rate_m3_s = float(leak_estimate["leak_rate_m3_s"])
        leak_rate_err = float(leak_estimate.get("leak_rate_err_m3_s", 0.0))
        delta_volume_m3 = leak_rate_m3_s * interval_dt
        delta_volume_err = leak_rate_err * interval_dt
        cumulative_volume_m3 += delta_volume_m3
        cumulative_volume_var += delta_volume_err ** 2
        cumulative_volume_err = float(np.sqrt(cumulative_volume_var))
        volume_series.append(
            {
                "t": current_t,
                "dt": interval_dt,
                "v_m3": delta_volume_m3,
                "v_m3_err": delta_volume_err,
                "v_lpm": leak_rate_m3_s * 60000.0,
                "v_lpm_err": leak_rate_err * 60000.0,
                "q_m3_s": leak_rate_m3_s,
                "q_m3_s_err": leak_rate_err,
                "meta": {
                    "ambient_rn_bq_m3": float(current_ambient),
                    "inventory_equiv_m3": float(leak_estimate["inventory_equiv_m3"]),
                    "leak_rate_raw_m3_s": float(leak_estimate["leak_rate_raw_m3_s"]),
                    "mean_activity_bq": float(leak_estimate["mean_activity_bq"]),
                    "start_activity_bq": float(leak_estimate["start_activity_bq"]),
                    "end_activity_bq": float(leak_estimate["end_activity_bq"]),
                    "method": leak_estimate["method"],
                    "clipped_to_zero": bool(leak_estimate["clipped_to_zero"]),
                },
            }
        )
        cumulative_series.append({
            "t": current_t,
            "v_m3_cum": cumulative_volume_m3,
            "v_m3_cum_err": cumulative_volume_err,
        })
        # Propagate end-activity uncertainty for next interval's start_activity.
        # end = start * survival + L * ambient * response
        # sigma_end ≈ sigma_start * survival  (leak rate contribution is correlated)
        survival = float(np.exp(-rn_decay_constant * interval_dt))
        previous_end_activity = float(leak_estimate["end_activity_bq"])
        previous_end_sigma = sigma_start * survival

    if negative_leak_rate_intervals_clipped:
        logger.info(
            "Clipped %d negative leak-rate intervals to 0.0 m^3/s",
            negative_leak_rate_intervals_clipped,
        )

    meta = {
        "source_isotopes": available_isotopes,
        "source_weights": weights,
        "detection_efficiency": {
            iso: float(detection_map.get(iso)) for iso in available_isotopes
        },
        "transport_efficiency": transport_eff,
        "retention_efficiency": retention_eff,
        "chain_correction": chain_correction,
        "rn222_half_life_s": rn_half_life_s,
        "rn222_decay_constant_s": rn_decay_constant,
        "volume_method": "interval_average_inventory_balance",
        "volume_units": "m^3 leaked per interval",
        "negative_leak_rate_intervals_clipped": negative_leak_rate_intervals_clipped,
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
