"""Plotting helpers for radon inference outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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

    parts: list[str] = []

    # Source isotopes
    isotopes = meta.get("source_isotopes")
    if isotopes:
        iso_list = ", ".join(str(iso) for iso in isotopes)
        parts.append(f"Source isotopes: {iso_list}")

    # Detection efficiencies
    detection = meta.get("detection_efficiency") or {}
    if isinstance(detection, Mapping):
        det_text = ", ".join(f"{iso}: {val:.3g}" for iso, val in detection.items())
        if det_text:
            parts.append(f"Detection eff: {det_text}")

    # Transport and retention
    transport = meta.get("transport_efficiency")
    retention = meta.get("retention_efficiency")
    if transport is not None:
        parts.append(f"Transport eff: {transport}")
    if retention is not None:
        parts.append(f"Retention eff: {retention}")

    # Chain correction
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
    if not isinstance(radon_results, Mapping):
        logger.info("radon_inference missing, skipping plot.")
        return

    series = radon_results.get("rn_inferred")
    if not series:
        logger.info("radon_inference missing, skipping plot.")
        return

    times, values = _extract_series(series, "rn_bq")
    if not times or not values:
        logger.info("No inferred radon data available for plotting")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    times_mpl = guard_mpl_times(times=times)
    label = _legend_label(radon_results.get("meta"))
    ax.plot(times_mpl, values, marker="o", linestyle="None", label=label)
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
    if not isinstance(radon_results, Mapping):
        logger.info("No radon results available for ambient radon plotting")
        return

    ambient_series = radon_results.get("ambient_rn")
    times, values = _extract_series(ambient_series, "rn_bq_per_m3")
    if not times or not values:
        logger.info("No ambient radon data available for plotting")
        return

    # Check if we need to trim to match rn_inferred length
    rn_inferred_series = radon_results.get("rn_inferred")
    rn_times, _ = _extract_series(rn_inferred_series, "rn_bq")
    if rn_times and len(times) != len(rn_times):
        min_len = min(len(times), len(rn_times))
        logger.info(
            "Trimming ambient_rn from %d to %d points to match rn_inferred length",
            len(times),
            min_len,
        )
        times = times[:min_len]
        values = values[:min_len]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if values are constant
    is_constant = False
    if len(set(values)) == 1:
        is_constant = True
        logger.info("Ambient radon is constant at %.3g Bq/m³", values[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    times_mpl = guard_mpl_times(times=times)
    ax.plot(times_mpl, values, marker="o", linestyle="None", color="#1f77b4")
    ax.set_ylabel("Ambient radon [Bq/m³]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)

    # Annotate if constant
    if is_constant:
        ax.text(
            0.98,
            0.98,
            f"Constant: {values[0]:.3g} Bq/m³",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8),
            fontsize="small",
        )

    fig.tight_layout()
    fig.savefig(out_dir / "ambient_radon.png", dpi=300)
    plt.close(fig)


def plot_volume_equiv_vs_time(
    radon_results: Mapping[str, object],
    out_dir: Path,
    volume_units: str | None = None,
) -> None:
    if not isinstance(radon_results, Mapping):
        logger.info("No radon results available for volume plotting")
        return

    volume_series = radon_results.get("volume_equiv")
    times, volumes_m3 = _extract_series(volume_series, "v_m3")
    _, volumes_lpm = _extract_series(volume_series, "v_lpm")
    if not times or not volumes_m3:
        logger.info("No equivalent volume data available for plotting")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine units from config or use default
    if volume_units is None:
        meta = radon_results.get("meta")
        if isinstance(meta, Mapping):
            volume_units = meta.get("volume_units", "m³ per interval")
        else:
            volume_units = "m³ per interval"

    times_mpl = guard_mpl_times(times=times)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(times_mpl, volumes_m3, marker="o", linestyle="None", color="#2ca02c")
    ax1.set_ylabel(f"Equivalent volume [{volume_units}]")
    ax1.set_xlabel("Time (UTC)")
    ax1.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax1, times_mpl)
    ax1.yaxis.get_offset_text().set_visible(False)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "equivalent_volume.png", dpi=300)
    plt.close(fig)

    # Create an additional plot in liters per interval for convenience
    volumes_liters = [v * 1000.0 for v in volumes_m3]
    liters_units = "L per interval"
    fig_l, ax_l = plt.subplots(figsize=(8, 5))
    ax_l.plot(times_mpl, volumes_liters, marker="o", linestyle="None", color="#17becf")
    ax_l.set_ylabel(f"Equivalent volume [{liters_units}]")
    ax_l.set_xlabel("Time (UTC)")
    ax_l.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax_l, times_mpl)
    ax_l.yaxis.get_offset_text().set_visible(False)

    fig_l.autofmt_xdate()
    fig_l.tight_layout()
    fig_l.savefig(out_dir / "equivalent_volume_liters.png", dpi=300)
    plt.close(fig_l)

    if volumes_lpm:
        paired_len = min(len(times_mpl), len(volumes_lpm))
        if paired_len == 0:
            logger.info("Equivalent flow data present but no overlapping timestamps; skipping flow plot")
        else:
            times_flow = times_mpl[:paired_len]
            volumes_flow = volumes_lpm[:paired_len]
            fig_flow, ax_flow = plt.subplots(figsize=(8, 5))
            ax_flow.plot(times_flow, volumes_flow, marker="o", linestyle="None", color="#ff7f0e")
            ax_flow.set_ylabel("Equivalent flow [L/min]")
            ax_flow.set_xlabel("Time (UTC)")
            ax_flow.ticklabel_format(axis="y", style="plain")
            setup_time_axis(ax_flow, times_flow)
            ax_flow.yaxis.get_offset_text().set_visible(False)
            fig_flow.autofmt_xdate()
            fig_flow.tight_layout()
            fig_flow.savefig(out_dir / "equivalent_flow.png", dpi=300)
            plt.close(fig_flow)

    # Only create cumulative plot if data is present
    cumulative_series = radon_results.get("volume_cumulative")
    if cumulative_series:
        cum_times, cum_values = _extract_series(cumulative_series, "v_m3_cum")
        if cum_times and cum_values:
            fig_cum, ax_cum = plt.subplots(figsize=(8, 5))
            times_mpl_cum = guard_mpl_times(times=cum_times)
            ax_cum.plot(times_mpl_cum, cum_values, marker="o", linestyle="None", color="#9467bd")
            ax_cum.set_ylabel(f"Cumulative volume [{volume_units}]")
            ax_cum.set_xlabel("Time (UTC)")
            ax_cum.ticklabel_format(axis="y", style="plain")
            setup_time_axis(ax_cum, times_mpl_cum)
            fig_cum.autofmt_xdate()
            ax_cum.yaxis.get_offset_text().set_visible(False)
            fig_cum.tight_layout()
            fig_cum.savefig(out_dir / "equivalent_volume_cumulative.png", dpi=300)
            plt.close(fig_cum)

            # Also provide a liters-per-interval cumulative view
            cum_values_liters = [val * 1000.0 for val in cum_values]
            fig_cum_l, ax_cum_l = plt.subplots(figsize=(8, 5))
            ax_cum_l.plot(
                times_mpl_cum,
                cum_values_liters,
                marker="o",
                linestyle="None",
                color="#8c564b",
            )
            ax_cum_l.set_ylabel("Cumulative volume [L per interval]")
            ax_cum_l.set_xlabel("Time (UTC)")
            ax_cum_l.ticklabel_format(axis="y", style="plain")
            setup_time_axis(ax_cum_l, times_mpl_cum)
            fig_cum_l.autofmt_xdate()
            ax_cum_l.yaxis.get_offset_text().set_visible(False)
            fig_cum_l.tight_layout()
            fig_cum_l.savefig(out_dir / "equivalent_volume_cumulative_liters.png", dpi=300)
            plt.close(fig_cum_l)
        else:
            logger.info("No valid cumulative volume data for plotting")
    else:
        logger.info("No cumulative volume series available, skipping cumulative plot")


__all__ = [
    "plot_rn_inferred_vs_time",
    "plot_ambient_rn_vs_time",
    "plot_volume_equiv_vs_time",
]

