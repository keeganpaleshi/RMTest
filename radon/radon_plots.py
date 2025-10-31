"""Plotting helpers for radon inference results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import logging

import matplotlib.pyplot as plt

from plot_utils._time_utils import guard_mpl_times, setup_time_axis


logger = logging.getLogger(__name__)

__all__ = [
    "plot_rn_inferred_vs_time",
    "plot_ambient_rn_vs_time",
    "plot_equivalent_volume_vs_time",
]


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_series(series: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None, key: str) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []
    if not series:
        return times, values
    for entry in series:
        t_val = _coerce_float(entry.get("t"))
        v_val = _coerce_float(entry.get(key))
        if t_val is None or v_val is None:
            continue
        times.append(t_val)
        values.append(v_val)
    return times, values


def plot_rn_inferred_vs_time(results: Mapping[str, Any], out_dir: Path | str) -> None:
    series = results.get("rn_inferred") if isinstance(results, Mapping) else None
    times, activities = _extract_series(series, "rn_bq")
    if not times or not activities:
        logger.info("Skipping rn_inferred plot: no data available")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    times_mpl = guard_mpl_times(times=times)
    ax.plot(times_mpl, activities, marker="o", linestyle="-", label="Inferred Rn-222")
    ax.set_ylabel("Activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)

    meta = results.get("meta", {}) if isinstance(results, Mapping) else {}
    legend_items = []
    if isinstance(meta, Mapping):
        weights = meta.get("source_weights")
        if isinstance(weights, Mapping) and weights:
            legend_items.append(
                "Weights: "
                + ", ".join(f"{iso}={weight:.3f}" for iso, weight in sorted(weights.items()))
            )
        detection = meta.get("detection_efficiency")
        if isinstance(detection, Mapping) and detection:
            legend_items.append(
                "Detection eff: "
                + ", ".join(
                    f"{iso}={eff:.3f}" for iso, eff in sorted(detection.items()) if eff is not None
                )
            )
        for label_key in ("transport_efficiency", "retention_efficiency"):
            val = _coerce_float(meta.get(label_key))
            if val is not None:
                legend_items.append(f"{label_key.replace('_', ' ').title()}: {val:.3f}")
        chain = meta.get("chain_correction")
        if chain:
            legend_items.append(f"Chain correction: {chain}")

    if legend_items:
        ax.text(
            0.01,
            0.02,
            "\n".join(legend_items),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path / "rn_inferred.png", dpi=300)
    plt.close(fig)


def plot_ambient_rn_vs_time(results: Mapping[str, Any], out_dir: Path | str) -> None:
    series = results.get("ambient_rn") if isinstance(results, Mapping) else None
    times, ambient = _extract_series(series, "rn_bq_per_m3")
    if not times or not ambient:
        logger.info("Skipping ambient radon plot: no data available")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    times_mpl = guard_mpl_times(times=times)
    ax.step(times_mpl, ambient, where="mid", label="Ambient radon")
    ax.set_ylabel("Radon concentration [Bq m$^{-3}$]")
    ax.set_xlabel("Time (UTC)")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path / "ambient_rn.png", dpi=300)
    plt.close(fig)


def plot_equivalent_volume_vs_time(results: Mapping[str, Any], out_dir: Path | str) -> None:
    series = results.get("volume_equiv") if isinstance(results, Mapping) else None
    if not series:
        logger.info("Skipping equivalent volume plot: no data available")
        return

    times: list[float] = []
    volumes_m3: list[float | None] = []
    volumes_lpm: list[float | None] = []
    for entry in series:
        t_val = _coerce_float(entry.get("t"))
        if t_val is None:
            continue
        times.append(t_val)
        volumes_m3.append(_coerce_float(entry.get("v_m3")))
        volumes_lpm.append(_coerce_float(entry.get("v_lpm")))

    if not times:
        logger.info("Skipping equivalent volume plot: timestamps missing")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    times_mpl = guard_mpl_times(times=times)

    plotted = False
    m3_values = [v for v in volumes_m3 if v is not None]
    if m3_values:
        ax1.plot(times_mpl, [v if v is not None else float("nan") for v in volumes_m3], label="Volume per interval")
        ax1.set_ylabel("Sampled volume [m$^3$]")
        plotted = True
    else:
        ax1.set_ylabel("Sampled volume [m$^3$]")

    ax1.set_xlabel("Time (UTC)")
    setup_time_axis(ax1, times_mpl)
    fig.autofmt_xdate()
    ax1.yaxis.get_offset_text().set_visible(False)

    lpm_values = [v for v in volumes_lpm if v is not None]
    if lpm_values:
        ax2 = ax1.twinx()
        ax2.plot(
            times_mpl,
            [v if v is not None else float("nan") for v in volumes_lpm],
            color="tab:orange",
            label="Equivalent flow",
        )
        ax2.set_ylabel("Equivalent flow [L min$^{-1}$]")
        ax2.yaxis.get_offset_text().set_visible(False)
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles or handles2:
            ax1.legend(handles + handles2, labels + labels2, loc="best")
        plotted = True
    elif plotted:
        ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path / "equivalent_volume.png", dpi=300)
    plt.close(fig)

    cumulative = results.get("volume_cumulative") if isinstance(results, Mapping) else None
    times_cum, volumes_cum = _extract_series(cumulative, "v_m3_cum")
    if times_cum and volumes_cum:
        fig_cum, ax_cum = plt.subplots(figsize=(8, 4.5))
        times_cum_mpl = guard_mpl_times(times=times_cum)
        ax_cum.plot(times_cum_mpl, volumes_cum, marker="o")
        ax_cum.set_ylabel("Cumulative volume [m$^3$]")
        ax_cum.set_xlabel("Time (UTC)")
        setup_time_axis(ax_cum, times_cum_mpl)
        fig_cum.autofmt_xdate()
        ax_cum.yaxis.get_offset_text().set_visible(False)
        fig_cum.tight_layout()
        fig_cum.savefig(out_path / "equivalent_volume_cumulative.png", dpi=300)
        plt.close(fig_cum)

