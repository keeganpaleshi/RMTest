"""Convert isotope count series into inferred radon activity."""

from __future__ import annotations

from collections import defaultdict
import logging
import math
from typing import Any, Iterable, Mapping

from radon.external_rn_loader import load_external_rn_series


logger = logging.getLogger(__name__)


def _normalize_isotope_list(
    isotope_series: Mapping[str, Iterable[Mapping[str, Any]]],
    source_isotopes: list[str],
) -> list[str]:
    available = []
    missing = []
    for iso in source_isotopes:
        series = isotope_series.get(iso)
        if series:
            available.append(iso)
        else:
            missing.append(iso)
    if missing:
        logger.warning(
            "radon inference: missing isotope series for %s", ", ".join(sorted(missing))
        )
    if available:
        return available
    logger.warning("radon inference: no isotope series available; skipping stage")
    return []


def _normalise_weights(weights: Mapping[str, float], isotopes: Iterable[str]) -> dict[str, float]:
    usable = {iso: float(weights.get(iso, 0.0)) for iso in isotopes}
    usable = {iso: w for iso, w in usable.items() if math.isfinite(w) and w > 0.0}
    if not usable:
        if isinstance(isotopes, list):
            n = len(isotopes)
        else:
            n = sum(1 for _ in isotopes)
        return {iso: 1.0 / n for iso in isotopes}
    total = sum(usable.values())
    if total <= 0:
        return {iso: 1.0 / len(usable) for iso in usable}
    return {iso: w / total for iso, w in usable.items()}


def _prepare_external_series(
    radon_cfg: Mapping[str, Any],
    timestamps: list[float],
    provided: Iterable | None,
) -> dict[float, float]:
    series: Iterable
    if provided is not None:
        series = provided
    else:
        external_cfg = radon_cfg.get("external_rn") if isinstance(radon_cfg, Mapping) else None
        if external_cfg:
            try:
                series = load_external_rn_series(external_cfg, timestamps)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("failed to load external radon series: %s", exc)
                series = []
        else:
            series = []

    output: dict[float, float] = {}
    for entry in series:
        if isinstance(entry, Mapping):
            ts = entry.get("t") or entry.get("time")
            val = entry.get("rn_bq_per_m3") or entry.get("value") or entry.get("rn")
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            ts, val = entry[0], entry[1]
        else:
            continue
        try:
            if hasattr(ts, "timestamp"):
                ts_f = float(ts.timestamp())
            else:
                ts_f = float(ts)
            val_f = float(val)
        except (TypeError, ValueError):
            continue
        output[ts_f] = val_f
    return output


def run_radon_inference(
    isotope_series: Mapping[str, Iterable[Mapping[str, Any]]],
    config: Mapping[str, Any] | None,
    external_rn_series: Iterable | None = None,
):
    """Return inferred radon activity and equivalent volume series.

    Parameters
    ----------
    isotope_series
        Mapping of isotope name to iterable of measurement dictionaries. Each
        dictionary should contain ``"t"`` (Unix seconds), ``"counts"`` and
        ``"dt"`` (bin live-time in seconds).
    config
        Full run configuration containing the ``radon_inference`` section.
    external_rn_series
        Optional iterable providing ambient radon measurements already aligned
        with the timestamps. When omitted the configuration is consulted and
        :func:`radon.external_rn_loader.load_external_rn_series` is used.
    """

    cfg = config or {}
    radon_cfg = cfg.get("radon_inference") if isinstance(cfg, Mapping) else None
    if not isinstance(radon_cfg, Mapping) or not radon_cfg.get("enabled", False):
        return None

    source_isotopes = list(radon_cfg.get("source_isotopes", []))
    if not source_isotopes:
        source_isotopes = sorted(str(k) for k in isotope_series.keys())

    available_isotopes = _normalize_isotope_list(isotope_series, source_isotopes)
    if not available_isotopes:
        return None

    weights_cfg = radon_cfg.get("source_weights", {})
    if not isinstance(weights_cfg, Mapping):
        weights_cfg = {}
    weights = _normalise_weights(weights_cfg, available_isotopes)

    det_eff_cfg = radon_cfg.get("detection_efficiency", {})
    if not isinstance(det_eff_cfg, Mapping):
        det_eff_cfg = {}
    transport_eff = float(radon_cfg.get("transport_efficiency", 1.0) or 1.0)
    retention_eff = float(radon_cfg.get("retention_efficiency", 1.0) or 1.0)
    if transport_eff <= 0:
        logger.warning("radon inference transport_efficiency must be positive; using 1.0")
        transport_eff = 1.0
    if retention_eff <= 0:
        logger.warning("radon inference retention_efficiency must be positive; using 1.0")
        retention_eff = 1.0
    eff_total = transport_eff * retention_eff

    chain_mode = str(radon_cfg.get("chain_correction", "none")).lower()
    supported_chain_modes = {"none", "assume_equilibrium"}
    if chain_mode not in supported_chain_modes:
        logger.warning(
            "radon inference chain_correction=%s not supported; treating as 'none'",
            chain_mode,
        )
        chain_mode = "none"

    aggregated: dict[float, dict[str, Any]] = defaultdict(lambda: {"dt": None, "components": []})
    for iso in available_isotopes:
        entries = isotope_series.get(iso, [])
        det_eff = det_eff_cfg.get(iso)
        try:
            det_eff_val = float(det_eff)
        except (TypeError, ValueError):
            det_eff_val = None
        if det_eff_val is None or det_eff_val <= 0:
            logger.warning(
                "radon inference: detection efficiency for %s must be positive", iso
            )
            continue

        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            try:
                t_val = float(entry.get("t"))
                counts = float(entry.get("counts", 0.0))
                dt = float(entry.get("dt"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(t_val) or not math.isfinite(dt) or dt <= 0:
                continue
            if not math.isfinite(counts):
                counts = 0.0

            activity = counts / (det_eff_val * dt) if det_eff_val > 0 else 0.0

            agg = aggregated[t_val]
            if agg["dt"] is None:
                agg["dt"] = dt
            else:
                if abs(float(agg["dt"]) - dt) > 1e-9:
                    agg["dt"] = float(agg["dt"])
            agg["components"].append({"isotope": iso, "activity": activity})

    if not aggregated:
        logger.warning("radon inference: no valid bins after processing inputs")
        return None

    timestamps_sorted = sorted(aggregated.keys())
    ambient_map = _prepare_external_series(radon_cfg, timestamps_sorted, external_rn_series)

    rn_series: list[dict[str, Any]] = []
    ambient_series: list[dict[str, Any]] = []
    volume_series: list[dict[str, Any]] = []
    volume_cumulative: list[dict[str, Any]] = []
    cumulative_volume = 0.0

    for t_val in timestamps_sorted:
        bin_info = aggregated[t_val]
        dt = float(bin_info.get("dt") or 0.0)
        if dt <= 0:
            continue
        components = bin_info.get("components", [])
        if not components:
            continue

        present_weights = {
            comp["isotope"]: weights.get(comp["isotope"])
            for comp in components
            if weights.get(comp["isotope"]) is not None
        }
        present_weights = {
            iso: w for iso, w in present_weights.items() if math.isfinite(w) and w > 0
        }
        if not present_weights:
            continue
        norm = sum(present_weights.values())
        if norm <= 0:
            continue

        activity = 0.0
        sources = []
        for comp in components:
            iso = comp.get("isotope")
            weight = present_weights.get(iso)
            if weight is None:
                continue
            frac = weight / norm
            act_val = float(comp.get("activity", 0.0))
            activity += frac * act_val
            sources.append(iso)

        if chain_mode != "none":
            # Placeholder for future corrections; behaviour matches "none".
            pass

        activity_adjusted = activity / eff_total if eff_total > 0 else activity

        rn_series.append({
            "t": t_val,
            "dt": dt,
            "rn_bq": activity_adjusted,
            "sources": sorted(set(sources)),
        })

        ambient_val = ambient_map.get(t_val)
        if ambient_val is not None and math.isfinite(ambient_val):
            ambient_series.append({"t": t_val, "rn_bq_per_m3": ambient_val})
            if ambient_val > 0:
                volume_m3 = activity_adjusted * dt / ambient_val
                if math.isfinite(volume_m3):
                    volume_lpm = volume_m3 * 1000.0 * 60.0 / dt
                    volume_entry = {
                        "t": t_val,
                        "v_m3": volume_m3,
                        "v_lpm": volume_lpm,
                        "dt": dt,
                    }
                    volume_series.append(volume_entry)
                    cumulative_volume += volume_m3
                    volume_cumulative.append({"t": t_val, "v_m3_cum": cumulative_volume})
        else:
            ambient_series.append({"t": t_val, "rn_bq_per_m3": None})

    if not rn_series:
        logger.warning("radon inference: no radon activity points computed")
        return None

    meta = {
        "source_isotopes": available_isotopes,
        "source_weights": weights,
        "detection_efficiency": {
            iso: float(det_eff_cfg.get(iso))
            for iso in available_isotopes
            if isinstance(det_eff_cfg, Mapping) and det_eff_cfg.get(iso) is not None
        },
        "transport_efficiency": transport_eff,
        "retention_efficiency": retention_eff,
        "chain_correction": chain_mode,
    }

    result = {
        "rn_inferred": rn_series,
        "ambient_rn": ambient_series,
        "volume_equiv": volume_series,
        "volume_cumulative": volume_cumulative,
        "meta": meta,
    }
    return result


__all__ = ["run_radon_inference"]

