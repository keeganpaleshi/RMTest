# -----------------------------------------------------
# plot_utils.py
# -----------------------------------------------------

import os
import numpy as np
import matplotlib as _mpl

_mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime
from pathlib import Path
from color_schemes import COLOR_SCHEMES
from constants import PO214, PO218, PO210
from .paths import get_targets

# Half-life constants used for the time-series overlay [seconds]
PO214_HALF_LIFE_S = PO214.half_life_s
PO218_HALF_LIFE_S = PO218.half_life_s

__all__ = [
    "extract_time_series",
    "plot_time_series",
    "plot_spectrum",
    "plot_equivalent_air",
    "plot_modeled_radon_activity",
    "plot_radon_activity",
    "plot_radon_trend",
    "plot_radon_activity_full",
    "plot_radon_trend_full",
]


def extract_time_series(timestamps, energies, window, t_start, t_end, bin_width_s=1.0):
    """Return histogram counts for events within an energy window.

    Parameters
    ----------
    timestamps : array-like
        Absolute event times in seconds.
    energies : array-like
        Event energies in MeV.
    window : tuple(float, float) or None
        Inclusive energy range. If ``None`` an empty array is returned.
    t_start, t_end : float
        Start and end times for the histogram.
    bin_width_s : float, optional
        Width of each time bin in seconds.

    Returns
    -------
    counts : np.ndarray
        Counts in each time bin.
    edges : np.ndarray
        Bin edges relative to ``t_start``.
    """

    if window is None:
        return np.array([]), np.array([])

    lo, hi = window
    timestamps = np.asarray(timestamps, dtype=float)
    energies = np.asarray(energies, dtype=float)

    mask = (
        (energies >= lo)
        & (energies <= hi)
        & (timestamps >= float(t_start))
        & (timestamps <= float(t_end))
    )

    rel_times = timestamps[mask] - float(t_start)

    n_bins = int(np.floor((float(t_end) - float(t_start)) / float(bin_width_s)))
    if n_bins < 1:
        n_bins = 1
    edges = np.arange(0, (n_bins + 1) * float(bin_width_s), float(bin_width_s))
    counts, _ = np.histogram(rel_times, bins=edges)
    return counts, edges


def plot_time_series(
    all_timestamps,
    all_energies,
    fit_results,
    t_start,
    t_end,
    config,
    out_png,
    hl_po214=None,
    hl_po218=None,
    *,
    model_errors=None,
    **_legacy_kwargs,
):
    """
    all_timestamps: 1D np.ndarray of absolute UNIX times (s)
    all_energies:   1D np.ndarray of energies (MeV)
    fit_results:    dict from fit_time_series(...)
    t_start, t_end: floats (absolute UNIX times) for the fit window
    config:         JSON dict or nested configuration
    out_png:        output path for the PNG file
    hl_po214, hl_po218: optional half-life values in seconds. If not
        provided, these are looked up using the configuration keys
        ``hl_po214`` and ``hl_po218`` under ``time_fit`` and default to
        ``PO214_HALF_LIFE_S`` and ``PO218_HALF_LIFE_S`` respectively.
        When Po-210 is plotted the overlay uses the ``hl_po210``
        configuration value.
    model_errors : dict[str, array-like], optional
        Mapping of isotope name to 1D arrays of uncertainties for the
        model curve. When provided, ``fill_between`` is used to draw
        +/-1 sigma bands around the corresponding model.
    """

    if fit_results is None:
        fit_results = {}

    # Convert timestamps to UNIX seconds when datetime64 or datetime objects
    ts_array = np.asarray(all_timestamps)
    if np.issubdtype(ts_array.dtype, "datetime64"):
        all_timestamps = ts_array.astype("int64") / 1e9
    elif np.issubdtype(ts_array.dtype, np.object_):
        if ts_array.size > 0 and isinstance(ts_array.flat[0], datetime):
            all_timestamps = np.array([dt.timestamp() for dt in ts_array], dtype=float)
        else:
            all_timestamps = ts_array.astype(float)
    else:
        all_timestamps = ts_array.astype(float)

    if isinstance(t_start, datetime):
        t_start = t_start.timestamp()
    elif isinstance(t_start, np.datetime64):
        t_start = float(t_start.astype("int64") / 1e9)
    if isinstance(t_end, datetime):
        t_end = t_end.timestamp()
    elif isinstance(t_end, np.datetime64):
        t_end = float(t_end.astype("int64") / 1e9)

    def _cfg_get(cfg, key, default=None):
        """Lookup ``key`` in ``cfg``.

        The search first checks the ``time_fit`` sub-dictionary, then the
        top level of ``cfg``.  Configuration keys should use lowercase
        names such as ``hl_po214`` and ``hl_po218``.
        """

        if not isinstance(cfg, dict):
            return default

        sub = cfg.get("time_fit", {})
        if isinstance(sub, dict) and key in sub:
            return sub[key]

        return cfg.get(key, default)

    default_const = config.get("nuclide_constants", {})
    default214 = default_const.get("Po214", PO214).half_life_s
    default218 = default_const.get("Po218", PO218).half_life_s

    if hl_po214 is None and "hl_po214" in _legacy_kwargs:
        hl_po214 = _legacy_kwargs.pop("hl_po214")
    if hl_po218 is None and "hl_po218" in _legacy_kwargs:
        hl_po218 = _legacy_kwargs.pop("hl_po218")

    def _hl_param(name, default):
        val = _cfg_get(config, name, default)
        if isinstance(val, list):
            return float(val[0]) if val else float(default)
        return float(default) if val is None else float(val)

    po214_hl = float(hl_po214) if hl_po214 is not None else _hl_param("hl_po214", default214)
    po218_hl = float(hl_po218) if hl_po218 is not None else _hl_param("hl_po218", default218)

    if po214_hl <= 0:
        raise ValueError("hl_po214 must be positive")
    if po218_hl <= 0:
        raise ValueError("hl_po218 must be positive")

    iso_params = {
        "Po210": {
            # Energy window for Po-210 events (optional)
            "window": _cfg_get(config, "window_po210"),
            "eff": float(_cfg_get(config, "eff_po210", [1.0])[0]),
            "half_life": _hl_param(
                "hl_po210",
                default_const.get("Po210", PO210).half_life_s,
            ),
        },
        "Po218": {
            # Energy window for Po-218 events
            "window": _cfg_get(config, "window_po218"),
            "eff": float(_cfg_get(config, "eff_po218", [1.0])[0]),
            "half_life": po218_hl,
        },
        "Po214": {
            # Energy window for Po-214 events
            "window": _cfg_get(config, "window_po214"),
            "eff": float(_cfg_get(config, "eff_po214", [1.0])[0]),
            "half_life": po214_hl,
        },
    }
    iso_list = [iso for iso, p in iso_params.items() if p["window"] is not None]
    # Time since t_start:
    times_rel = all_timestamps - t_start

    # 1) Choose binning:
    # Newer config files may store plot options under keys prefixed with
    # "plot_".  Fall back to the old names for backwards compatibility.
    bin_mode = str(
        config.get(
            "plot_time_binning_mode",
            config.get("time_bin_mode", "fixed"),
        )
    ).lower()
    if bin_mode in ("fd", "auto"):
        # Freedman Diaconis rule on the entire time range:
        data = times_rel[(times_rel >= 0) & (times_rel <= (t_end - t_start))]
        if len(data) < 2:
            n_bins = 1
        else:
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            if iqr <= 0:
                n_bins = int(config.get("time_bins_fallback", 1))
            else:
                bin_width = 2 * iqr / (len(data) ** (1.0 / 3.0))
                if isinstance(bin_width, np.timedelta64):
                    bin_width = bin_width / np.timedelta64(1, "s")
                    data_range = (data.max() - data.min()) / np.timedelta64(1, "s")
                else:
                    data_range = data.max() - data.min()
                n_bins = max(1, int(np.ceil(data_range / float(bin_width))))
    else:
        # fixed-width bins (integer-second data) – use floor so the
        # very last partial bin is dropped and every remaining bin has
        # exactly the same width.
        dt = int(
            config.get(
                "plot_time_bin_width_s",
                config.get("time_bin_s", 3600),
            )
        )
        n_bins = int(np.floor((t_end - t_start) / dt))
        if n_bins < 1:
            n_bins = 1

    # ------------------------------------------------------------------
    # Build equally-spaced edges so delta t is identical for each bin
    # ------------------------------------------------------------------
    if bin_mode not in ("fd", "auto"):
        edges = np.arange(0, (n_bins + 1) * dt, dt, dtype=float)
    else:
        edges = np.linspace(0, (t_end - t_start), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    centers_abs = t_start + centers
    centers_dt = mdates.date2num([datetime.utcfromtimestamp(t) for t in centers_abs])
    bin_widths = np.diff(edges)

    # Optional normalisation to counts / s (set in config)
    normalise_rate = bool(config.get("plot_time_normalise_rate", False))

    # 2) Plot each isotope s histogram + overlay the model:
    plt.figure(figsize=(8, 6))
    palette_name = str(config.get("palette", "default"))
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    colors = {
        "Po214": palette.get("Po214", "#d62728"),
        "Po218": palette.get("Po218", "#1f77b4"),
        "Po210": palette.get("Po210", "#2ca02c"),
    }

    for iso in iso_list:
        emin, emax = iso_params[iso]["window"]
        mask_iso = (
            (all_energies >= emin)
            & (all_energies <= emax)
            & (all_timestamps >= t_start)
            & (all_timestamps <= t_end)
        )
        t_iso_rel = times_rel[mask_iso]

        # Histogram of observed counts:
        counts_iso, _ = np.histogram(t_iso_rel, bins=edges)
        if normalise_rate:
            counts_iso = counts_iso / bin_widths

        style = str(config.get("plot_time_style", "steps")).lower()
        if style == "lines":
            plt.plot(
                centers_dt,
                counts_iso,
                marker="o",
                linestyle="-",
                color=colors[iso],
                label=f"Data {iso}",
            )
        else:
            plt.step(
                centers_dt,
                counts_iso,
                where="mid",
                color=colors[iso],
                label=f"Data {iso}",
            )

        # Overlay the continuous model curve (scaled to counts/bin)
        # only when fit results are provided for this isotope.
        has_fit = any(k in fit_results for k in (f"E_{iso}", "E"))
        if has_fit:
            lam = np.log(2.0) / iso_params[iso]["half_life"]
            eff = iso_params[iso]["eff"]

            E_iso = fit_results.get(f"E_{iso}", fit_results.get("E", 0.0))
            B_iso = fit_results.get(f"B_{iso}", fit_results.get("B", 0.0))
            N0_iso = fit_results.get(f"N0_{iso}", fit_results.get("N0", 0.0))

            # r_iso(t_rel) = eff * [E*(1 - exp(-lam*t_rel)) + lam*N0*exp(-lam*t_rel)] + B
            r_rel = (
                eff
                * (
                    E_iso * (1.0 - np.exp(-lam * centers))
                    + lam * N0_iso * np.exp(-lam * centers)
                )
                + B_iso
            )

            # Convert rate (counts/s) to expected counts per bin if not normalising
            model_counts = r_rel if normalise_rate else r_rel * bin_widths
            plt.plot(
                centers_dt,
                model_counts,
                color=colors[iso],
                lw=2,
                ls="--",
                label=f"Model {iso}",
            )
            if model_errors and iso in model_errors:
                err = np.asarray(model_errors[iso], dtype=float)
                if err.size == model_counts.size:
                    kw = {"step": "mid"} if style != "lines" else {}
                    plt.fill_between(
                        centers_dt,
                        model_counts - err,
                        model_counts + err,
                        color=colors[iso],
                        alpha=0.3,
                        **kw,
                    )
                else:
                    raise ValueError("model_errors array length mismatch")

    plt.xlabel("Time")
    plt.ylabel("Counts / s" if normalise_rate else "Counts per bin")
    title_isos = " & ".join(iso_list)
    plt.title(f"{title_isos} Time Series Fit")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize="small")

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:  # older matplotlib
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    targets = get_targets(config, out_png)
    for p in targets.values():
        plt.savefig(p, dpi=300)
    plt.close()

    # (Optionally) also write a small JSON of the binned values:
    if config.get("dump_time_series_json", False):
        import json

        ts_summary = {"centers_s": centers.tolist(), "widths_s": bin_widths.tolist()}
        for iso in iso_list:
            emin, emax = iso_params[iso]["window"]
            ts_summary[f"counts_{iso}"] = np.histogram(
                times_rel[
                    (all_energies >= emin)
                    & (all_energies <= emax)
                    & (all_timestamps >= t_start)
                    & (all_timestamps <= t_end)
                ],
                bins=edges,
            )[0].tolist()
            ts_summary[f"live_time_{iso}_s"] = bin_widths.tolist()
            ts_summary[f"eff_{iso}"] = [iso_params[iso]["eff"]] * len(bin_widths)
        base = Path(out_png).with_suffix("")
        json_path = base.with_name(base.name + "_ts.json")
        with open(json_path, "w") as jf:
            json.dump(ts_summary, jf, indent=2)


def _model_counts_per_bin(edges: np.ndarray, fit_vals: dict[str, float]):
    """Return model prediction in counts per bin."""

    width = np.diff(edges)
    centers = edges[:-1] + width / 2.0
    sigma_E = fit_vals.get("sigma_E", 1.0)
    b0 = fit_vals.get("b0", 0.0)
    b1 = fit_vals.get("b1", 0.0)
    S_bkg = fit_vals.get("S_bkg", 0.0)
    bkg_cent = b0 + b1 * centers
    bkg_norm = b0 * (edges[-1] - edges[0]) + 0.5 * b1 * (edges[-1] ** 2 - edges[0] ** 2)
    lam = np.zeros_like(centers)
    if bkg_norm > 0:
        lam += S_bkg * bkg_cent / bkg_norm
    for pk in ("Po210", "Po218", "Po214"):
        mu_key = f"mu_{pk}"
        amp_key = f"S_{pk}"
        if mu_key in fit_vals and amp_key in fit_vals:
            mu = fit_vals[mu_key]
            amp = fit_vals[amp_key]
            lam += (
                amp
                / (sigma_E * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((centers - mu) / sigma_E) ** 2)
            )
    return centers, lam * width

def plot_spectrum(
    energies,
    fit_vals=None,
    out_png="spectrum.png",
    bins=400,
    bin_edges=None,
    config=None,
):
    """Plot energy spectrum and optional fit overlay.

    Parameters
    ----------
    energies : array-like
        Energy values in MeV.
    fit_vals : dict, optional
        Dictionary of fit parameters to overlay. If provided, a
        residual panel is also produced.
    out_png : str, optional
        Output path (extension used if ``plot_save_formats`` not set).
    bins : int, optional
        Number of bins if ``bin_edges`` is not supplied.
    bin_edges : array-like, optional
        Explicit, strictly increasing bin edges in MeV.  Non-uniform widths
        are supported and override ``bins``.
    config : dict, optional
        Plotting configuration dictionary.
    """
    show_res = bool(fit_vals)
    if (
        bin_edges is None
        and config is not None
        and "plot_spectrum_binsize_adc" in config
    ):
        step = float(config["plot_spectrum_binsize_adc"])
        e_min, e_max = energies.min(), energies.max()
        bin_edges = np.arange(e_min, e_max + step, step)

    if bin_edges is not None:
        hist, edges = np.histogram(energies, bins=bin_edges)
    else:
        hist, edges = np.histogram(energies, bins=bins)

    width = np.diff(edges)
    centers = edges[:-1] + width / 2.0

    if show_res:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_main = plt.subplots(figsize=(8, 6))
        ax_res = None

    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    hist_color = palette.get("hist", "#808080")
    ax_main.bar(centers, hist, width=width, color=hist_color, alpha=0.7, label="Data")

    # If an explicit Po-210 window is provided, focus the x-axis on that region
    win_p210 = None
    if config is not None:
        win_p210 = config.get("window_po210")
    if win_p210 is not None:
        lo, hi = win_p210
        ax_main.set_xlim(lo, hi)

    if fit_vals:
        centers, model_counts = _model_counts_per_bin(edges, fit_vals)
        palette_name = str(config.get("palette", "default")) if config else "default"
        palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
        fit_color = palette.get("fit", "#ff0000")
        ax_main.plot(centers, model_counts, color=fit_color, lw=2, label="Fit")

        if show_res:
            residuals = hist - model_counts
            ax_res.bar(
                centers,
                residuals,
                width=width,
                color=hist_color,
                alpha=0.7,
            )
            ax_res.axhline(0.0, color="#000000", lw=1)
            ax_res.set_ylabel("Residuals")

    ax_main.set_ylabel("Counts per bin")
    ax_main.set_title("Energy Spectrum")
    if fit_vals:
        ax_main.legend(fontsize="small")
    if ax_res is not None:
        ax_res.set_xlabel("Energy (MeV)")
    else:
        ax_main.set_xlabel("Energy (MeV)")
    fig.tight_layout()
    targets = get_targets(config, out_png)
    for p in targets.values():
        fig.savefig(p, dpi=300)
    plt.close(fig)

    return ax_main


def plot_radon_activity_full(times, activity, errors, out_png, config=None):
    """Plot radon activity versus time with uncertainties."""
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)
    errors = np.asarray(errors, dtype=float)

    times_dt = mdates.date2num([datetime.utcfromtimestamp(t) for t in times])

    plt.figure(figsize=(8, 4))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("radon_activity", "#9467bd")
    plt.errorbar(times_dt, activity, yerr=errors, fmt="o-", color=color)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Radon Activity (Bq)")
    plt.title("Extrapolated Radon Activity vs. Time")

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    base_dt = times_dt[0]

    def _to_seconds(x):
        return (x - base_dt) * 86400.0

    def _to_dates(x):
        return base_dt + x / 86400.0

    secax = ax.secondary_xaxis("top", functions=(_to_seconds, _to_dates))

    def _sec_formatter(x, pos=None):
        h = x / 3600.0
        if h >= 1:
            if abs(h - round(h)) < 1e-6:
                return f"{int(x)} s ({int(h)} h)"
            return f"{int(x)} s ({h:g} h)"
        return f"{int(x)} s"

    secax.xaxis.set_major_formatter(mticker.FuncFormatter(_sec_formatter))
    secax.set_xlabel("Elapsed Time (s)")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    targets = get_targets(config, out_png)
    for p in targets.values():
        plt.savefig(p, dpi=300)
    plt.close()


def plot_equivalent_air(times, volumes, errors, conc, out_png, config=None):
    """Plot equivalent air volume versus time.

    Parameters
    ----------
    conc : float or str or None
        Ambient concentration label to include in the plot title. When ``None``
        the concentration is omitted from the title.
    """
    times = np.asarray(times, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    times_dt = mdates.date2num([datetime.utcfromtimestamp(t) for t in times])

    plt.figure(figsize=(8, 4))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("equivalent_air", "#2ca02c")
    plt.errorbar(times_dt, volumes, yerr=errors, fmt="o-", color=color)
    plt.xlabel("Time")
    plt.ylabel("Equivalent Air Volume")
    if conc is None:
        title = "Equivalent Air Volume vs. Time"
    else:
        title = f"Equivalent Air Volume vs. Time (ambient {conc} Bq/L)"
    plt.title(title)

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    targets = get_targets(config, out_png)
    for p in targets.values():
        plt.savefig(p, dpi=300)
    plt.close()


def plot_modeled_radon_activity(
    times,
    E,
    dE,
    N0,
    dN0,
    half_life_s,
    out_png,
    config=None,
):
    """Compute and plot modeled radon activity over time."""
    from radon_activity import radon_activity_curve

    activity, sigma = radon_activity_curve(times, E, dE, N0, dN0, half_life_s)
    plot_radon_activity_full(times, activity, sigma, out_png, config=config)


def plot_radon_trend_full(times, activity, out_png, config=None):
    """Plot modeled radon activity trend without uncertainties."""
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)

    times_dt = mdates.date2num([datetime.utcfromtimestamp(t) for t in times])

    plt.figure(figsize=(8, 4))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("radon_activity", "#9467bd")
    plt.plot(times_dt, activity, "o-", color=color)
    plt.xlabel("Time")
    plt.ylabel("Radon Activity (Bq)")
    plt.title("Radon Activity Trend")

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    targets = get_targets(config, out_png)
    for p in targets.values():
        plt.savefig(p, dpi=300)
    plt.close()


def plot_radon_activity(ts_dict, outdir):
    """Simple wrapper to plot radon activity time series."""
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    t = ts_dict["time"]
    y = ts_dict["activity"]
    e = ts_dict["error"]
    plt.errorbar(t, y, yerr=e, fmt="o")
    plt.ylabel("Radon activity [Bq]")
    plt.xlabel("Time (UTC)")
    plt.tight_layout()
    plt.savefig(outdir / "radon_activity.png", dpi=300)
    plt.close()


def plot_radon_trend(ts_dict, outdir):
    """Simple wrapper to plot a radon activity trend."""
    import numpy as np
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    t = np.asarray(ts_dict["time"])
    y = np.asarray(ts_dict["activity"])
    coeff = np.polyfit(t, y, 1)
    plt.plot(t, y, "o")
    plt.plot(t, np.polyval(coeff, t))
    plt.ylabel("Radon activity [Bq]")
    plt.xlabel("Time (UTC)")
    plt.tight_layout()
    plt.savefig(outdir / "radon_trend.png", dpi=300)
    plt.close()


def plot_spectrum_comparison(
    pre_energies,
    post_energies,
    *,
    bins=400,
    bin_edges=None,
    out_png="spectrum_pre_post.png",
    config=None,
):
    """Overlay spectra before and after filtering and return ROI differences."""

    pre = np.asarray(pre_energies, dtype=float)
    post = np.asarray(post_energies, dtype=float)
    if bin_edges is None:
        data = np.concatenate([pre, post])
        if data.size == 0:
            bin_edges = np.linspace(0.0, 1.0, bins + 1)
        else:
            bin_edges = np.histogram_bin_edges(data, bins=bins)

    hist_pre, edges = np.histogram(pre, bins=bin_edges)
    hist_post, _ = np.histogram(post, bins=edges)

    width = np.diff(edges)
    centers = edges[:-1] + width / 2.0

    fig, ax = plt.subplots(figsize=(8, 6))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color_pre = palette.get("hist", "#808080")
    color_post = palette.get("fit", "#ff0000")
    ax.bar(centers, hist_pre, width=width, color=color_pre, alpha=0.5, label="Pre")
    ax.bar(centers, hist_post, width=width, color=color_post, alpha=0.5, label="Post")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Counts per bin")
    ax.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

    roi_diff = {}
    if config is not None:
        for iso in ("Po210", "Po218", "Po214"):
            win = config.get(f"window_{iso.lower()}")
            if win is None:
                continue
            c_pre = int(((pre >= win[0]) & (pre <= win[1])).sum())
            c_post = int(((post >= win[0]) & (post <= win[1])).sum())
            roi_diff[iso] = c_post - c_pre

    return roi_diff


def plot_activity_grid(result_map, out_png="burst_scan.png", config=None):
    """Visualise radon activity on a parameter grid."""

    if not result_map:
        return

    mults = sorted({m for m, _ in result_map})
    wins = sorted({w for _, w in result_map})
    grid = np.empty((len(mults), len(wins)))
    for i, m in enumerate(mults):
        for j, w in enumerate(wins):
            grid[i, j] = result_map.get((m, w), np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(wins), max(wins), min(mults), max(mults)],
    )
    ax.set_xlabel("burst_window_size_s")
    ax.set_ylabel("burst_multiplier")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Radon Activity (Bq)")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# -----------------------------------------------------
# End of plot_utils.py
# -----------------------------------------------------
