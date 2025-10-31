"""Plotting helpers for radon inference outputs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import guard_mpl_times, setup_time_axis
from plot_utils.paths import get_targets
from utils.time_utils import parse_timestamp


def _series_to_arrays(series: Iterable[Mapping[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    times = []
    values = []
    for entry in series or []:
        if not isinstance(entry, Mapping):
            continue
        ts_val = entry.get("t") or entry.get("timestamp")
        if ts_val is None:
            continue
        try:
            ts = parse_timestamp(ts_val).to_pydatetime()
        except Exception:
            continue
        val = entry.get(key)
        if val is None:
            continue
        try:
            val_f = float(val)
        except (TypeError, ValueError):
            continue
        times.append(ts)
        values.append(val_f)
    if not times:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=float)
    return np.asarray(times), np.asarray(values, dtype=float)


def _save_figure(out_path, config: Mapping[str, Any] | None) -> None:
    targets = get_targets(config, out_path)
    for target in targets.values():
        plt.savefig(target, dpi=300)
    plt.close()


def plot_rn_inferred_vs_time(results: Mapping[str, Any], out_path, *, config=None) -> None:
    """Plot inferred Rn-222 activity over time."""

    series = results.get("rn_inferred") if isinstance(results, Mapping) else None
    times, values = _series_to_arrays(series or [], "rn_bq")
    if times.size == 0:
        return

    meta = results.get("meta", {}) if isinstance(results, Mapping) else {}
    weights = meta.get("resolved_source_weights", {})
    detection = meta.get("detection_efficiency", {})
    transport = meta.get("transport_efficiency")
    retention = meta.get("retention_efficiency")
    chain = meta.get("chain_correction")

    plt.figure(figsize=(8, 4.5))
    plt.plot(guard_mpl_times(times), values, marker="o", linestyle="-", label="Inferred Rn222")
    plt.ylabel("Activity [Bq]")
    plt.title("Rn222 inferred from daughters")
    if weights:
        weight_text = ", ".join(f"{iso}: {w:.3f}" for iso, w in sorted(weights.items()))
        plt.legend(title=f"Weights ({weight_text})", loc="best", fontsize="small")
    else:
        plt.legend(loc="best", fontsize="small")

    setup_time_axis(plt.gca(), guard_mpl_times(times))
    plt.gcf().autofmt_xdate()

    footer_parts = []
    if detection:
        det_text = ", ".join(f"{iso}: {val:.3f}" for iso, val in sorted(detection.items()))
        footer_parts.append(f"Detection eff: {det_text}")
    if transport is not None:
        footer_parts.append(f"Transport eff: {transport:.3f}")
    if retention is not None:
        footer_parts.append(f"Retention eff: {retention:.3f}")
    if chain:
        footer_parts.append(f"Chain correction: {chain}")
    if footer_parts:
        plt.figtext(0.5, 0.01, " | ".join(footer_parts), ha="center", fontsize="x-small")

    _save_figure(out_path, config)


def plot_ambient_rn_vs_time(results: Mapping[str, Any], out_path, *, config=None) -> None:
    """Plot ambient mine radon concentration over time."""

    series = results.get("ambient_rn") if isinstance(results, Mapping) else None
    times, values = _series_to_arrays(series or [], "rn_bq_per_m3")
    if times.size == 0:
        return

    plt.figure(figsize=(8, 4.5))
    plt.step(guard_mpl_times(times), values, where="mid", label="Ambient Rn", color="#1f77b4")
    plt.ylabel("Concentration [Bq/m³]")
    plt.title("Ambient mine radon (input)")
    plt.legend(loc="best", fontsize="small")
    setup_time_axis(plt.gca(), guard_mpl_times(times))
    plt.gcf().autofmt_xdate()

    _save_figure(out_path, config)


def plot_volume_equiv_vs_time(results: Mapping[str, Any], out_path, *, config=None) -> None:
    """Plot equivalent sampled volume and cumulative volume."""

    series = results.get("volume_equiv") if isinstance(results, Mapping) else None
    cumulative = (
        results.get("volume_cumulative") if isinstance(results, Mapping) else None
    )

    times_v, values_m3 = _series_to_arrays(series or [], "v_m3")
    _, values_lpm = _series_to_arrays(series or [], "v_lpm")
    times_c, values_c = _series_to_arrays(cumulative or [], "v_m3_cum")

    if times_v.size == 0 and times_c.size == 0:
        return

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    ax_flow = axes[0]
    if times_v.size > 0:
        x_vals = guard_mpl_times(times_v)
        if np.isfinite(values_lpm).any():
            ax_flow.plot(x_vals, values_lpm, marker="o", linestyle="-", color="#d62728")
            ax_flow.set_ylabel("Equivalent flow [L/min]")
        else:
            ax_flow.plot(x_vals, values_m3, marker="o", linestyle="-", color="#d62728")
            ax_flow.set_ylabel("Equivalent volume [m³]")
    ax_flow.set_title("Equivalent sampled volume")

    ax_cum = axes[1]
    if times_c.size > 0:
        ax_cum.plot(
            guard_mpl_times(times_c),
            values_c,
            marker="o",
            linestyle="-",
            color="#2ca02c",
        )
    ax_cum.set_ylabel("Cumulative volume [m³]")
    ax_cum.set_xlabel("Time (UTC)")

    for ax in axes:
        base = times_c if ax is ax_cum and times_c.size > 0 else times_v
        if base.size == 0 and times_c.size > 0:
            base = times_c
        if base.size > 0:
            setup_time_axis(ax, guard_mpl_times(base))
        ax.tick_params(axis="x", labelrotation=30)

    fig.tight_layout()
    _save_figure(out_path, config)


__all__ = [
    "plot_rn_inferred_vs_time",
    "plot_ambient_rn_vs_time",
    "plot_volume_equiv_vs_time",
]

