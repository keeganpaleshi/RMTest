# -----------------------------------------------------
# plot_utils.py
# -----------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from color_schemes import COLOR_SCHEMES
from constants import PO214, PO218, PO210

# Half-life constants used for the time-series overlay [seconds]
PO214_HALF_LIFE_S = PO214.half_life_s
PO218_HALF_LIFE_S = PO218.half_life_s

__all__ = [
    "plot_time_series",
    "plot_spectrum",
    "plot_radon_activity",
    "plot_equivalent_air",
    "plot_modeled_radon_activity",
    "plot_radon_trend",
]


def plot_time_series(
    all_timestamps,
    all_energies,
    fit_results,
    t_start,
    t_end,
    config,
    out_png,
    hl_Po214=None,
    hl_Po218=None,
):
    """
    all_timestamps: 1D np.ndarray of absolute UNIX times (s)
    all_energies:   1D np.ndarray of energies (MeV)
    fit_results:    dict from fit_time_series(...)
    t_start, t_end: floats (absolute UNIX times) for the fit window
    config:         JSON dict or nested configuration
    out_png:        output path for the PNG file
    hl_Po214, hl_Po218: optional half-life values in seconds. If not
        provided, these are looked up in ``config`` and default to
        ``PO214_HALF_LIFE_S`` and ``PO218_HALF_LIFE_S`` respectively.
    """

    if fit_results is None:
        fit_results = {}

    def _cfg_get(cfg, key, default=None):
        if isinstance(cfg, dict) and "time_fit" in cfg and key in cfg["time_fit"]:
            return cfg["time_fit"][key]
        if isinstance(cfg, dict) and key in cfg:
            return cfg[key]
        return default

    default_const = config.get("nuclide_constants", {})
    default214 = default_const.get("Po214", PO214).half_life_s
    default218 = default_const.get("Po218", PO218).half_life_s

    po214_hl = (
        float(hl_Po214)
        if hl_Po214 is not None
        else float(_cfg_get(config, "hl_Po214", [default214])[0])
    )
    po218_hl = (
        float(hl_Po218)
        if hl_Po218 is not None
        else float(_cfg_get(config, "hl_Po218", [default218])[0])
    )

    if po214_hl <= 0:
        raise ValueError("hl_Po214 must be positive")
    if po218_hl <= 0:
        raise ValueError("hl_Po218 must be positive")

    iso_params = {
        "Po214": {
            "window": _cfg_get(config, "window_Po214"),
            "eff": float(_cfg_get(config, "eff_Po214", [1.0])[0]),
            "half_life": po214_hl,
        },
        "Po218": {
            "window": _cfg_get(config, "window_Po218"),
            "eff": float(_cfg_get(config, "eff_Po218", [1.0])[0]),
            "half_life": po218_hl,
        },
        "Po210": {
            "window": _cfg_get(config, "window_Po210"),
            "eff": float(_cfg_get(config, "eff_Po210", [1.0])[0]),
            "half_life": float(
                _cfg_get(
                    config,
                    "hl_Po210",
                    [default_const.get("Po210", PO210).half_life_s],
                )[0]
            ),
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
    # Build equally-spaced edges so Δt is identical for each bin
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
        "Po214": palette.get("Po214", "tab:red"),
        "Po218": palette.get("Po218", "tab:blue"),
        "Po210": palette.get("Po210", "tab:green"),
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

    plt.xlabel("Time")
    plt.ylabel("Counts / s" if normalise_rate else "Counts per bin")
    title_isos = " & ".join(iso_list)
    plt.title(f"{title_isos} Time Series Fit")
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
    dirpath = os.path.dirname(out_png) or "."
    os.makedirs(dirpath, exist_ok=True)

    # Determine which formats to save. If not specified, fall back to the
    # extension of the provided output path.
    fmt_default = os.path.splitext(out_png)[1].lstrip(".") or "png"
    save_fmts = config.get("plot_save_formats", [fmt_default])
    if isinstance(save_fmts, str):
        save_fmts = [save_fmts]

    base = os.path.splitext(out_png)[0]
    for fmt in save_fmts:
        out_file = base + f".{fmt}"
        plt.savefig(out_file, dpi=300)
    plt.close()

    # (Optionally) also write a small JSON of the binned values:
    if config.get("dump_time_series_json", False):
        import json

        ts_summary = {"centers_s": centers.tolist()}
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
        with open(out_png.replace(".png", "_ts.json"), "w") as jf:
            json.dump(ts_summary, jf, indent=2)


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
        Explicit bin edges in MeV.  Overrides ``bins``.
    config : dict, optional
        Plotting configuration dictionary.
    """
    show_res = bool(fit_vals)
    if bin_edges is None and config is not None and "plot_spectrum_binsize_adc" in config:
        step = float(config["plot_spectrum_binsize_adc"])
        e_min, e_max = energies.min(), energies.max()
        bin_edges = np.arange(e_min, e_max + step, step)

    if bin_edges is not None:
        hist, edges = np.histogram(energies, bins=bin_edges)
    else:
        hist, edges = np.histogram(energies, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    if show_res:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6),
            gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_main = plt.subplots(figsize=(8, 6))
        ax_res = None

    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    hist_color = palette.get("hist", "gray")
    ax_main.bar(centers, hist, width=width, color=hist_color, alpha=0.7, label="Data")

    # If an explicit Po-210 window is provided, focus the x-axis on that region
    win_p210 = None
    if config is not None:
        win_p210 = config.get("window_Po210")
    if win_p210 is not None:
        lo, hi = win_p210
        ax_main.set_xlim(lo, hi)

    if fit_vals:
        x = np.linspace(edges[0], edges[-1], 1000)
        sigma_E = fit_vals.get("sigma_E", 1.0)
        y = fit_vals.get("b0", 0.0) + fit_vals.get("b1", 0.0) * x
        for pk in ("Po210", "Po218", "Po214"):
            mu_key = f"mu_{pk}"
            amp_key = f"S_{pk}"
            if mu_key in fit_vals and amp_key in fit_vals:
                mu = fit_vals[mu_key]
                amp = fit_vals[amp_key]
                y += amp / (sigma_E * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma_E) ** 2)
        palette_name = str(config.get("palette", "default")) if config else "default"
        palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
        fit_color = palette.get("fit", "red")
        ax_main.plot(x, y * width, color=fit_color, lw=2, label="Fit")

        if show_res:
            y_cent = fit_vals.get("b0", 0.0) + fit_vals.get("b1", 0.0) * centers
            for pk in ("Po210", "Po218", "Po214"):
                mu_key = f"mu_{pk}"
                amp_key = f"S_{pk}"
                if mu_key in fit_vals and amp_key in fit_vals:
                    mu = fit_vals[mu_key]
                    amp = fit_vals[amp_key]
                    y_cent += amp / (
                        sigma_E * np.sqrt(2 * np.pi)
                    ) * np.exp(-0.5 * ((centers - mu) / sigma_E) ** 2)
            model_counts = y_cent * width
            residuals = hist - model_counts
            ax_res.bar(
                centers,
                residuals,
                width=width,
                color=hist_color,
                alpha=0.7,
            )
            ax_res.axhline(0.0, color="black", lw=1)
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
    dirpath = os.path.dirname(out_png) or "."
    os.makedirs(dirpath, exist_ok=True)

    fmt_default = os.path.splitext(out_png)[1].lstrip(".") or "png"
    save_fmts = []
    if config is not None:
        save_fmts = config.get("plot_save_formats", [])
    if not save_fmts:
        save_fmts = [fmt_default]
    if isinstance(save_fmts, str):
        save_fmts = [save_fmts]

    base = os.path.splitext(out_png)[0]
    for fmt in save_fmts:
        fig.savefig(base + f".{fmt}", dpi=300)
    plt.close(fig)

    return ax_main


def plot_radon_activity(times, activity, errors, out_png, config=None):
    """Plot radon activity versus time with uncertainties."""
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)
    errors = np.asarray(errors, dtype=float)

    times_dt = mdates.date2num([datetime.utcfromtimestamp(t) for t in times])

    plt.figure(figsize=(8, 4))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("radon_activity", "tab:purple")
    plt.errorbar(times_dt, activity, yerr=errors, fmt="o-", color=color)
    plt.xlabel("Time")
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
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    dirpath = os.path.dirname(out_png) or "."
    os.makedirs(dirpath, exist_ok=True)

    fmt_default = os.path.splitext(out_png)[1].lstrip(".") or "png"
    fmts = config.get("plot_save_formats", [fmt_default]) if config else [fmt_default]
    if isinstance(fmts, str):
        fmts = [fmts]
    base = os.path.splitext(out_png)[0]
    for fmt in fmts:
        plt.savefig(base + f".{fmt}", dpi=300)
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
    color = palette.get("equivalent_air", "tab:green")
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
    dirpath = os.path.dirname(out_png) or "."
    os.makedirs(dirpath, exist_ok=True)

    fmt_default = os.path.splitext(out_png)[1].lstrip(".") or "png"
    fmts = config.get("plot_save_formats", [fmt_default]) if config else [fmt_default]
    if isinstance(fmts, str):
        fmts = [fmts]
    base = os.path.splitext(out_png)[0]
    for fmt in fmts:
        plt.savefig(base + f".{fmt}", dpi=300)
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
    plot_radon_activity(times, activity, sigma, out_png, config=config)


def plot_radon_trend(times, activity, out_png, config=None):
    """Plot modeled radon activity trend without uncertainties."""
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)

    times_dt = mdates.date2num([datetime.utcfromtimestamp(t) for t in times])

    plt.figure(figsize=(8, 4))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("radon_activity", "tab:purple")
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

    dirpath = os.path.dirname(out_png) or "."
    os.makedirs(dirpath, exist_ok=True)

    fmt_default = os.path.splitext(out_png)[1].lstrip(".") or "png"
    fmts = config.get("plot_save_formats", [fmt_default]) if config else [fmt_default]
    if isinstance(fmts, str):
        fmts = [fmts]
    base = os.path.splitext(out_png)[0]
    for fmt in fmts:
        plt.savefig(base + f".{fmt}", dpi=300)
    plt.close()


# -----------------------------------------------------
# End of plot_utils.py
# -----------------------------------------------------
