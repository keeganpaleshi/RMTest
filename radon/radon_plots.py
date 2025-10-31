"""Plotting helpers for radon inference outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from plot_utils._time_utils import guard_mpl_times, setup_time_axis

logger = logging.getLogger(__name__)


def _extract_series(
    data: Iterable[Mapping[str, object]] | None,
    value_key: str,
) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []
    if not data:
        return times, values
    for entry in data:
        try:
            t = float(entry.get("t"))
            val = float(entry.get(value_key))
        except (AttributeError, TypeError, ValueError):
            continue
        if not np.isfinite(t) or not np.isfinite(val):
            continue
        times.append(t)
        values.append(val)
    return times, values


def _format_meta(meta: Mapping[str, object] | None) -> str:
    if not isinstance(meta, Mapping):
        return ""

    detection = meta.get("detection_efficiency") or {}
    if isinstance(detection, Mapping):
        det_text = ", ".join(f"{iso}: {val:.3g}" for iso, val in detection.items())
    else:
        det_text = ""

    parts: list[str] = []
    if det_text:
        parts.append(f"Detection eff: {det_text}")
    transport = meta.get("transport_efficiency")
    retention = meta.get("retention_efficiency")
    if transport is not None:
        parts.append(f"Transport eff: {transport}")
    if retention is not None:
        parts.append(f"Retention eff: {retention}")
    chain = meta.get("chain_correction")
    if chain:
        parts.append(f"Chain correction: {chain}")
    return "\n".join(parts)


def _legend_label(meta: Mapping[str, object] | None) -> str:
    if not isinstance(meta, Mapping):
        return "Inferred Rn222"
    isotopes = meta.get("source_isotopes") or []
    weights = meta.get("source_weights") or {}
    if not isotopes:
        return "Inferred Rn222"

    labels = []
    for iso in isotopes:
        if isinstance(weights, Mapping) and iso in weights:
            labels.append(f"{iso} (w={weights[iso]:.2f})")
        else:
            labels.append(str(iso))
    return "Weighted by " + ", ".join(labels)


def plot_rn_inferred_vs_time(radon_results: Mapping[str, object], out_dir: Path) -> None:
    series = radon_results.get("rn_inferred") if isinstance(radon_results, Mapping) else None
    times, values = _extract_series(series, "rn_bq")
    if not times or not values:
        logger.info("No inferred radon data available for plotting")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    times_mpl = guard_mpl_times(times=times)
    label = _legend_label(radon_results.get("meta"))
    ax.plot(times_mpl, values, marker="o", linestyle="-", label=label)
    ax.set_ylabel("Rn222 activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    if label:
        ax.legend(loc="best", fontsize="small")

    meta_text = _format_meta(radon_results.get("meta"))
    if meta_text:
        fig.text(0.01, 0.01, meta_text, fontsize="x-small", va="bottom")

    fig.tight_layout()
    fig.savefig(out_dir / "radon_inferred.png", dpi=300)
    plt.close(fig)


def plot_ambient_rn_vs_time(radon_results: Mapping[str, object], out_dir: Path) -> None:
    ambient_series = radon_results.get("ambient_rn") if isinstance(radon_results, Mapping) else None
    times, values = _extract_series(ambient_series, "rn_bq_per_m3")
    if not times or not values:
        logger.info("No ambient radon data available for plotting")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    times_mpl = guard_mpl_times(times=times)
    ax.plot(times_mpl, values, marker="o", linestyle="-", color="#1f77b4")
    ax.set_ylabel("Ambient radon [Bq/m³]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "ambient_radon.png", dpi=300)
    plt.close(fig)


def plot_volume_equiv_vs_time(radon_results: Mapping[str, object], out_dir: Path) -> None:
    volume_series = radon_results.get("volume_equiv") if isinstance(radon_results, Mapping) else None
    times, volumes_m3 = _extract_series(volume_series, "v_m3")
    _, volumes_lpm = _extract_series(volume_series, "v_lpm")
    if not times or not volumes_m3:
        logger.info("No equivalent volume data available for plotting")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times_mpl = guard_mpl_times(times=times)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(times_mpl, volumes_m3, marker="o", linestyle="-", color="#2ca02c")
    ax1.set_ylabel("Equivalent volume [m³]")
    ax1.set_xlabel("Time (UTC)")
    ax1.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax1, times_mpl)
    ax1.yaxis.get_offset_text().set_visible(False)

    if volumes_lpm:
        ax2 = ax1.twinx()
        ax2.plot(times_mpl, volumes_lpm, marker="x", linestyle="--", color="#ff7f0e")
        ax2.set_ylabel("Equivalent flow [L/min]")
        ax2.ticklabel_format(axis="y", style="plain")
        ax2.yaxis.get_offset_text().set_visible(False)
        ax2.grid(False)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "equivalent_volume.png", dpi=300)
    plt.close(fig)

    cumulative_series = radon_results.get("volume_cumulative") if isinstance(radon_results, Mapping) else None
    cum_times, cum_values = _extract_series(cumulative_series, "v_m3_cum")
    if cum_times and cum_values:
        fig_cum, ax_cum = plt.subplots(figsize=(8, 5))
        times_mpl_cum = guard_mpl_times(times=cum_times)
        ax_cum.plot(times_mpl_cum, cum_values, marker="o", linestyle="-", color="#9467bd")
        ax_cum.set_ylabel("Cumulative volume [m³]")
        ax_cum.set_xlabel("Time (UTC)")
        ax_cum.ticklabel_format(axis="y", style="plain")
        setup_time_axis(ax_cum, times_mpl_cum)
        fig_cum.autofmt_xdate()
        ax_cum.yaxis.get_offset_text().set_visible(False)
        fig_cum.tight_layout()
        fig_cum.savefig(out_dir / "equivalent_volume_cumulative.png", dpi=300)
        plt.close(fig_cum)


__all__ = [
    "plot_rn_inferred_vs_time",
    "plot_ambient_rn_vs_time",
    "plot_volume_equiv_vs_time",
]

