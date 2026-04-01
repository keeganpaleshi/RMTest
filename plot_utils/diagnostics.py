"""Spectral fit diagnostic plots: correlation matrix, pull histogram, parameter table."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from scipy import fft as sp_fft

logger = logging.getLogger(__name__)

# Standard alpha-decay peak energies (MeV) for region definitions
_PEAK_REGIONS = {
    "Po210": (5.10, 5.50),
    "Po218": (5.80, 6.20),
    "Po214": (7.40, 7.90),
    "Po212": (8.55, 9.10),
}
_INTER_PEAK_REGIONS = {
    "gap_210_218": (5.50, 5.80),
    "gap_218_214": (6.20, 7.40),
    "gap_214_212": (7.90, 8.55),
}


# ═══════════════════════════════════════════════════════════════════════
# Fit-validation diagnostics (complementary to overfitting diagnostics)
# ═══════════════════════════════════════════════════════════════════════


def compute_fit_validation_diagnostics(
    fit_params: Mapping[str, object],
    min_counts: float = 5.0,
) -> dict:
    """Assess whether the spectral model adequately describes the data.

    While pull diagnostics focus on *overfitting* (are residuals
    suspiciously structured?), these metrics focus on *goodness-of-fit*
    (does the model actually describe the data?).

    Metrics
    -------
    Global fit quality
    ~~~~~~~~~~~~~~~~~~
    chi2_pvalue : p-value from chi2 CDF with ``ndf`` degrees of freedom.
        Very small values (< 0.001) indicate the model is rejected.
    deviance : sum of deviance residuals (2 * Σ[d·ln(d/m) - (d-m)]).
        The correct Poisson GoF statistic; converges to chi2(ndf) for
        large counts.  More accurate than Pearson chi2 for low-count bins.
    deviance_pvalue : p-value of the deviance statistic.
    baker_cousins : Baker-Cousins statistic = 2·Σ[m - d + d·ln(d/m)].
        Equivalent to -2·log(L_saturated/L_fit); the recommended GoF
        for Poisson binned maximum-likelihood fits.
    baker_cousins_pvalue : p-value from chi2(ndf).

    Per-peak quality
    ~~~~~~~~~~~~~~~~
    peak_<iso>_chi2 : Pearson chi2 in the peak window.
    peak_<iso>_ndf : number of bins in the peak window.
    peak_<iso>_chi2_ndf : ratio (should be ~1.0 for a good fit).
    peak_<iso>_deviance : deviance restricted to the peak window.
    peak_<iso>_pvalue : per-peak chi2 p-value.

    Parameter health
    ~~~~~~~~~~~~~~~~
    n_at_bound : number of free parameters at their bound limits.
    params_at_bound : list of parameter names hitting bounds.
    n_large_correlation : number of |correlation| > 0.95 pairs.
    max_abs_correlation : maximum off-diagonal |correlation|.
    """
    model = np.asarray(fit_params.get("_plot_model_total", []), dtype=float)
    hist = np.asarray(fit_params.get("_plot_hist", []), dtype=float)
    centers = np.asarray(fit_params.get("_plot_centers", []), dtype=float)
    if model.size == 0 or hist.size == 0:
        return {}

    mask = model > min_counts
    n = int(mask.sum())
    if n < 10:
        return {}

    d = hist[mask]
    m = model[mask]
    e = centers[mask] if centers.size == model.size else np.arange(n, dtype=float)

    ndf = int(fit_params.get("ndf", n - int(fit_params.get("n_free_params", 0))))
    if ndf < 1:
        ndf = max(1, n - int(fit_params.get("n_free_params", 0)))

    result: dict = {}

    # ── Global Pearson chi2 p-value ──────────────────────────────────
    chi2 = float(fit_params.get("chi2", np.sum((d - m) ** 2 / m)))
    result["chi2"] = round(chi2, 2)
    result["ndf"] = ndf
    result["chi2_ndf"] = round(chi2 / ndf, 4) if ndf > 0 else float("nan")
    result["chi2_pvalue"] = float(sp_stats.chi2.sf(chi2, ndf))

    # ── Deviance (Poisson GoF) ───────────────────────────────────────
    # G^2 = 2 * Σ[ d*ln(d/m) - (d - m) ]  for d > 0
    # For d == 0 bins: contribution is 2 * m
    pos = d > 0
    deviance = np.zeros_like(d)
    deviance[pos] = 2.0 * (d[pos] * np.log(d[pos] / m[pos]) - (d[pos] - m[pos]))
    deviance[~pos] = 2.0 * m[~pos]
    total_deviance = float(np.sum(deviance))
    result["deviance"] = round(total_deviance, 2)
    result["deviance_ndf"] = round(total_deviance / ndf, 4) if ndf > 0 else float("nan")
    result["deviance_pvalue"] = float(sp_stats.chi2.sf(total_deviance, ndf))

    # ── Baker-Cousins statistic ──────────────────────────────────────
    # C_min = 2 * Σ[ m - d + d*ln(d/m) ]  for d > 0
    # For d == 0: contribution is 2 * m
    bc = np.zeros_like(d)
    bc[pos] = 2.0 * (m[pos] - d[pos] + d[pos] * np.log(d[pos] / m[pos]))
    bc[~pos] = 2.0 * m[~pos]
    total_bc = float(np.sum(bc))
    result["baker_cousins"] = round(total_bc, 2)
    result["baker_cousins_ndf"] = round(total_bc / ndf, 4) if ndf > 0 else float("nan")
    result["baker_cousins_pvalue"] = float(sp_stats.chi2.sf(total_bc, ndf))

    # ── Per-peak chi2 and deviance ───────────────────────────────────
    n_free_per_peak = int(fit_params.get("n_free_params", 0))
    n_peaks = sum(
        1 for iso, (lo, hi) in _PEAK_REGIONS.items()
        if np.any((e >= lo) & (e <= hi))
    )
    # Rough per-peak free params: total free / n_peaks (approximate)
    k_per_peak = max(1, n_free_per_peak // max(n_peaks, 1))

    for iso, (lo, hi) in _PEAK_REGIONS.items():
        rmask = (e >= lo) & (e <= hi)
        nr = int(rmask.sum())
        if nr < 3:
            continue
        d_r = d[rmask]
        m_r = m[rmask]
        chi2_r = float(np.sum((d_r - m_r) ** 2 / np.maximum(m_r, 1.0)))
        ndf_r = max(1, nr - k_per_peak)

        pos_r = d_r > 0
        dev_r = np.zeros_like(d_r)
        dev_r[pos_r] = 2.0 * (d_r[pos_r] * np.log(d_r[pos_r] / m_r[pos_r])
                               - (d_r[pos_r] - m_r[pos_r]))
        dev_r[~pos_r] = 2.0 * m_r[~pos_r]

        result[f"peak_{iso}_chi2"] = round(chi2_r, 2)
        result[f"peak_{iso}_ndf"] = ndf_r
        result[f"peak_{iso}_chi2_ndf"] = round(chi2_r / ndf_r, 4)
        result[f"peak_{iso}_deviance"] = round(float(np.sum(dev_r)), 2)
        result[f"peak_{iso}_pvalue"] = float(sp_stats.chi2.sf(chi2_r, ndf_r))

    # ── Parameter-at-bound check ─────────────────────────────────────
    bounds = fit_params.get("_bounds", {})
    at_bound = []
    param_names = fit_params.get("param_names", [])
    if isinstance(param_names, (list, tuple)):
        for pname in param_names:
            val = fit_params.get(pname)
            err = fit_params.get(f"d{pname}", 0.0)
            if not isinstance(val, (int, float)) or not isinstance(err, (int, float)):
                continue
            if err <= 0:
                continue  # fixed parameter, not relevant
            # Check against explicit bounds if available
            b = bounds.get(pname)
            if isinstance(b, (list, tuple)) and len(b) == 2:
                lo_b, hi_b = b
                if isinstance(lo_b, (int, float)) and abs(val - lo_b) < 1e-6 * max(abs(lo_b), 1):
                    at_bound.append(pname)
                elif isinstance(hi_b, (int, float)) and abs(val - hi_b) < 1e-6 * max(abs(hi_b), 1):
                    at_bound.append(pname)
    result["n_at_bound"] = len(at_bound)
    result["params_at_bound"] = at_bound

    # ── Correlation matrix health ────────────────────────────────────
    cov = fit_params.get("correlation_matrix")
    if cov is None:
        cov = fit_params.get("cov")
    if cov is not None:
        try:
            cov_arr = np.asarray(cov, dtype=float)
            if cov_arr.ndim == 2:
                n_p = cov_arr.shape[0]
                # Normalise to correlation if not already (diagonal >> 1 → covariance)
                diag = np.diag(cov_arr)
                if np.any(diag > 1.5):
                    # This is a covariance matrix, convert to correlation
                    sd = np.sqrt(np.maximum(diag, 1e-30))
                    corr = cov_arr / np.outer(sd, sd)
                else:
                    corr = cov_arr
                # Mask diagonal and fixed params (zero row/col)
                np.fill_diagonal(corr, 0.0)
                abs_corr = np.abs(corr)
                result["max_abs_correlation"] = round(float(np.max(abs_corr)), 4)
                result["n_large_correlation"] = int(np.sum(abs_corr > 0.95) // 2)
        except Exception:
            pass

    return result


def plot_correlation_matrix(
    fit_result,
    out_dir: str | Path,
) -> None:
    """Plot the parameter correlation matrix as a heatmap.

    Fixed parameters (zero off-diagonal correlations) are filtered out so
    that the heatmap shows only the free-parameter submatrix.
    """
    out_dir = Path(out_dir)

    cov = getattr(fit_result, "cov", None)
    param_index = getattr(fit_result, "param_index", None)
    if cov is None or param_index is None or cov.size == 0:
        logger.info("No covariance matrix available; skipping correlation plot")
        return

    names = sorted(param_index, key=lambda k: param_index[k])
    n = len(names)
    if cov.shape != (n, n):
        logger.warning("Covariance shape %s vs %d params; skipping", cov.shape, n)
        return

    # Compute correlation matrix
    diag = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    corr = cov / np.outer(diag, diag)
    np.clip(corr, -1.0, 1.0, out=corr)

    # Identify free parameters: those with meaningful off-diagonal correlations
    # or non-trivial diagonal variance (fixed params have near-zero variance
    # relative to the prior width)
    free_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        off_diag = np.concatenate([corr[i, :i], corr[i, i + 1:]])
        if np.any(np.abs(off_diag) > 0.005) or diag[i] > 1e-10:
            free_mask[i] = True

    free_idx = np.where(free_mask)[0]
    if len(free_idx) < 2:
        logger.info("Fewer than 2 free params; skipping correlation plot")
        return

    corr_free = corr[np.ix_(free_idx, free_idx)]
    free_names = [names[i] for i in free_idx]
    nf = len(free_names)

    # Check whether we have meaningful off-diagonal content
    off_diag_mask = ~np.eye(nf, dtype=bool)
    has_off_diag = np.any(np.abs(corr_free[off_diag_mask]) > 0.01)

    short_names = [_short_name(nm) for nm in free_names]

    fig, ax = plt.subplots(figsize=(max(8, nf * 0.50), max(7, nf * 0.45)))
    im = ax.imshow(corr_free, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation coefficient")

    ax.set_xticks(range(nf))
    ax.set_yticks(range(nf))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    # Extra bottom margin so rotated x-labels aren't clipped
    fig.subplots_adjust(bottom=0.18)

    n_fixed = n - nf
    title = "Parameter Correlation Matrix"
    if has_off_diag:
        title += f" ({nf} free, {n_fixed} fixed)"
    else:
        title += " (diagonal only \u2014 Hesse unavailable)"
    ax.set_title(title, fontsize=11)

    # Annotate cells with values
    if nf <= 30:
        for i in range(nf):
            for j in range(nf):
                val = corr_free[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                # Bold for strong correlations
                weight = "bold" if abs(val) > 0.7 and i != j else "normal"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=color, fontweight=weight)

    # Add a text box listing strongest correlations
    pairs = []
    for i in range(nf):
        for j in range(i + 1, nf):
            r = corr_free[i, j]
            if abs(r) > 0.5:
                pairs.append((abs(r), r, short_names[i], short_names[j]))
    pairs.sort(reverse=True)
    if pairs:
        lines = ["Strong correlations (|r| > 0.5):"]
        for _, r, p1, p2 in pairs[:8]:
            lines.append(f"  {p1} \u2194 {p2}: {r:+.2f}")
        text = "\n".join(lines)
        ax.text(1.02, 0.02, text, transform=ax.transAxes, fontsize=6,
                va="bottom", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="gray", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_dir / "fit_correlation_matrix.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation matrix to %s", out_dir / "fit_correlation_matrix.png")

    # --- Relative error bar chart ---
    _plot_relative_errors(fit_result, out_dir)


def _plot_relative_errors(fit_result, out_dir: Path) -> None:
    """Horizontal bar chart of relative parameter uncertainties."""
    params = getattr(fit_result, "params", None)
    if params is None:
        return

    # Use MINOS errors to identify free (fitted) parameters.
    # Fixed params have Hessian errors that are numerical artifacts and
    # should be excluded from the relative errors plot.
    minos = getattr(fit_result, "minos_errors", None) or {}
    _free_names = set(minos.keys()) if minos else set()

    rows = []  # (name, rel_err_pct)
    skip = {"fit_valid", "likelihood_path", "aic", "nll", "chi2", "chi2_ndf",
            "ndf", "n_free_params", "ndf_eff", "chi2_ndf_eff",
            "ndf_effective", "chi2_ndf_effective",
            "covariance_method", "minos_method"}
    for key in sorted(params.keys()):
        if key.startswith("d") or key.startswith("_") or key in skip:
            continue
        if key.startswith("cov_"):
            continue
        # If we have MINOS info, only show free (fitted) parameters
        if _free_names and key not in _free_names:
            continue
        val = params.get(key)
        err = params.get("d" + key, 0.0)
        if not isinstance(val, (int, float)) or not isinstance(err, (int, float)):
            continue
        if err <= 0 or abs(val) < 1e-15:
            continue
        rel = 100.0 * err / abs(val)
        rows.append((_short_name(key), rel))

    if not rows:
        return

    rows.sort(key=lambda x: x[1])
    names_plot = [r[0] for r in rows]
    vals_plot = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.30 + 1)))
    colors = []
    for v in vals_plot:
        if v > 50:
            colors.append("#d32f2f")  # red
        elif v > 20:
            colors.append("#f57c00")  # orange
        elif v > 10:
            colors.append("#fbc02d")  # yellow
        else:
            colors.append("#388e3c")  # green
    # Use log scale if the range spans more than 10x to avoid
    # one huge bar squashing everything else
    use_log = max(vals_plot) > 10 * (np.median(vals_plot) + 1)

    bars = ax.barh(range(len(rows)), vals_plot, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names_plot, fontsize=7)
    ax.set_title("Parameter Relative Uncertainties", fontsize=11)

    if use_log:
        ax.set_xscale("log")
        ax.set_xlabel("Relative uncertainty (%, log scale)")
        ax.axvline(10, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(100, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax.set_xlabel("Relative uncertainty (%)")
        ax.axvline(10, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add value labels
    for bar, v in zip(bars, vals_plot):
        x_pos = bar.get_width() * (1.15 if use_log else 1.0) + (0 if use_log else 0.5)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(out_dir / "fit_relative_errors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved relative errors to %s", out_dir / "fit_relative_errors.png")


def plot_pull_histogram(
    fit_params: Mapping[str, object],
    out_dir: str | Path,
) -> None:
    """Plot a standalone histogram of pull values with Gaussian overlay."""
    out_dir = Path(out_dir)

    model = np.asarray(fit_params.get("_plot_model_total", []), dtype=float)
    hist = np.asarray(fit_params.get("_plot_hist", []), dtype=float)
    if model.size == 0 or hist.size == 0:
        logger.info("No stored model/hist; skipping pull histogram")
        return

    # Compute pulls
    mask = model > 5.0
    residuals = hist - model
    pulls = residuals[mask] / np.sqrt(model[mask])

    fig, ax = plt.subplots(figsize=(7, 5))
    n_bins = min(60, max(20, int(np.sqrt(len(pulls)))))
    counts, bin_edges, patches = ax.hist(
        pulls, bins=n_bins, density=True, color="#4daf4a", alpha=0.7,
        edgecolor="white", linewidth=0.3, label="Pulls",
    )

    # Overlay standard Gaussian
    x = np.linspace(-4.5, 4.5, 200)
    gauss = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
    ax.plot(x, gauss, "k--", linewidth=1.5, label="$\\mathcal{N}(0,1)$")

    # +-1 and +-2 sigma bands
    ax.axvspan(-1, 1, alpha=0.08, color="green")
    ax.axvspan(-2, 2, alpha=0.04, color="orange")

    # Stats annotation
    mu = np.mean(pulls)
    sigma = np.std(pulls)
    n_total = len(pulls)
    pct_gt2 = 100.0 * np.sum(np.abs(pulls) > 2) / n_total
    stats_text = (
        f"$N$ = {n_total}\n"
        f"$\\langle$pull$\\rangle$ = {mu:.3f}\n"
        f"$\\sigma$(pull) = {sigma:.3f}\n"
        f"|pull| > 2$\\sigma$: {pct_gt2:.1f}%"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    ax.set_xlabel("Pull ($\\sigma$)")
    ax.set_ylabel("Probability density")
    ax.set_title("Pull Distribution", fontsize=11)
    ax.set_xlim(-4.5, 4.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fit_pull_histogram.png", dpi=200)
    plt.close(fig)
    logger.info("Saved pull histogram to %s", out_dir / "fit_pull_histogram.png")


def plot_parameter_summary(
    fit_params: Mapping[str, object],
    out_dir: str | Path,
    minos_errors: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Plot a table of fitted parameters with values and uncertainties.

    If *minos_errors* is provided (``{name: (lower, upper)}``), an extra
    column shows the asymmetric MINOS confidence interval.
    """
    out_dir = Path(out_dir)

    # Collect parameters that have both value and error
    rows = []
    skip_prefixes = ("_plot_", "_dnl", "fit_valid", "aic", "ndf", "chi2")
    for key in sorted(fit_params.keys()):
        if key.startswith("d") or key.startswith("_") or key.startswith("F"):
            continue
        if any(key.startswith(p) for p in skip_prefixes):
            continue
        val = fit_params.get(key)
        err = fit_params.get("d" + key, 0.0)
        if not isinstance(val, (int, float)):
            continue
        if not isinstance(err, (int, float)):
            err = 0.0
        rows.append((key, float(val), float(err)))

    if not rows:
        logger.info("No parameters to summarise; skipping table")
        return

    has_minos = minos_errors is not None and len(minos_errors) > 0

    fig_width = 10 if has_minos else 8
    fig, ax = plt.subplots(figsize=(fig_width, max(4, len(rows) * 0.35 + 1.5)))
    ax.axis("off")

    if has_minos:
        col_labels = ["Parameter", "Value", "Sym. Err", "MINOS (lo/hi)", "Rel. Err (%)"]
    else:
        col_labels = ["Parameter", "Value", "Uncertainty", "Rel. Err (%)"]

    table_data = []
    for name, val, err in rows:
        rel = 100.0 * err / abs(val) if val != 0 and err > 0 else 0.0
        if has_minos:
            me = minos_errors.get(name)
            if me is not None:
                minos_str = f"{me[0]:+.4g} / {me[1]:+.4g}"
            else:
                minos_str = "--"
            table_data.append([
                _short_name(name),
                f"{val:.6g}",
                f"{err:.4g}" if err > 0 else "--",
                minos_str,
                f"{rel:.1f}" if err > 0 else "--",
            ])
        else:
            table_data.append([
                _short_name(name),
                f"{val:.6g}",
                f"{err:.4g}" if err > 0 else "--",
                f"{rel:.1f}" if err > 0 else "--",
            ])

    # Add chi2 and AIC if available
    chi2_ndf = fit_params.get("chi2_ndf")
    aic = fit_params.get("aic")
    n_extra = len(col_labels)
    if chi2_ndf is not None:
        row = ["chi2/NDF", f"{float(chi2_ndf):.3f}"] + [""] * (n_extra - 2)
        table_data.append(row)
    if aic is not None:
        row = ["AIC", f"{float(aic):.1f}"] + [""] * (n_extra - 2)
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)

    # Header row styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading + highlight poorly constrained params
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#D9E2F3")
        # Highlight rows with >50% relative error in red
        rel_col = len(col_labels) - 1
        try:
            rel_str = table_data[i - 1][rel_col]
            if rel_str and rel_str != "--":
                rel_val = float(rel_str)
                if rel_val > 50:
                    for j in range(len(col_labels)):
                        table[i, j].set_facecolor("#FFCDD2")
                elif rel_val > 30:
                    for j in range(len(col_labels)):
                        table[i, j].set_facecolor("#FFE0B2")
        except (ValueError, IndexError):
            pass

    title = "Spectral Fit Parameters"
    if has_minos:
        title += " (with MINOS errors)"
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fit_parameter_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved parameter table to %s", out_dir / "fit_parameter_table.png")


def plot_split_half_comparison(
    split_half_result: dict,
    out_dir: str | Path,
) -> None:
    """Horizontal bar chart of split-half z-scores per parameter.

    Parameters are categorised as *shape* (sigma, tau, f_shelf, f_halo, b),
    *amplitude* (S), or *position* (mu).  Amplitude and position params are
    expected to differ between time halves (activity drift, calibration shift),
    so the overfitting pass/fail is based on shape params only.
    """
    out_dir = Path(out_dir)
    params = split_half_result.get("parameters", [])
    if not params:
        logger.info("No split-half parameters; skipping comparison plot")
        return

    # Sort by category then by absolute z-score
    cat_order = {"shape": 0, "amplitude": 1, "position": 2}
    params_sorted = sorted(
        params,
        key=lambda p: (cat_order.get(p.get("category", "shape"), 0), abs(p["z_score"])),
    )
    names = [_short_name(p["name"]) for p in params_sorted]
    z_scores = [p["z_score"] for p in params_sorted]
    categories = [p.get("category", "shape") for p in params_sorted]

    # Color by category + z magnitude
    colors = []
    for z, cat in zip(z_scores, categories):
        if cat != "shape":
            colors.append("#9e9e9e")  # grey for amplitude/position
        elif abs(z) > 2.0:
            colors.append("#d32f2f")
        elif abs(z) > 1.0:
            colors.append("#f57c00")
        else:
            colors.append("#388e3c")

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.30 + 1.5)))
    bars = ax.barh(range(len(names)), z_scores, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("z-score (half A - half B)")

    # Reference lines
    for z_ref in [-2, -1, 1, 2]:
        ax.axvline(z_ref, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    # Shade bands
    ax.axvspan(-1, 1, alpha=0.06, color="green")
    ax.axvspan(-2, 2, alpha=0.03, color="orange")

    # Clip x-range so outlier bars don't squash the interesting region.
    # Add value annotations for any bars that extend past the limit.
    max_abs_z = max(abs(z) for z in z_scores) if z_scores else 3
    xlim = min(max_abs_z * 1.1, max(5.0, np.percentile(np.abs(z_scores), 90) * 1.5))
    ax.set_xlim(-xlim, xlim)
    for bar, z, name in zip(bars, z_scores, names):
        if abs(z) > xlim * 0.95:
            ax.text(xlim * 0.97 * np.sign(z), bar.get_y() + bar.get_height() / 2,
                    f" {z:+.1f}", va="center", ha="right" if z > 0 else "left",
                    fontsize=6, fontweight="bold")

    # Draw category separator line
    n_shape = sum(1 for c in categories if c == "shape")
    if 0 < n_shape < len(categories):
        ax.axhline(n_shape - 0.5, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.95, n_shape + 0.2,
                "\u2190 rate/position (expected drift)",
                fontsize=6, ha="right", va="bottom", color="#666666")

    max_z_shape = split_half_result.get("max_z_shape", 0)
    mean_z_shape = split_half_result.get("mean_abs_z_shape", 0)
    n_gt2 = split_half_result.get("n_shape_z_gt_2", 0)
    n_gt3 = split_half_result.get("n_shape_z_gt_3", 0)
    passed = split_half_result.get("pass", True)
    n_a = split_half_result.get("n_events_A", 0)
    n_b = split_half_result.get("n_events_B", 0)

    stats_text = (
        f"Events: {n_a} + {n_b}\n"
        f"Shape params only:\n"
        f"  max |z| = {max_z_shape:.2f}\n"
        f"  \u27e8|z|\u27e9 = {mean_z_shape:.2f}\n"
        f"  |z| > 2: {n_gt2}\n"
        f"  |z| > 3: {n_gt3}\n"
        f"{'PASS' if passed else 'WARN: possible overfit'}"
    )
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="#c8e6c9" if passed else "#ffcdd2",
                      ec="gray", alpha=0.9))

    ax.set_title("Split-Half Overfitting Validation", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fit_split_half.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved split-half comparison to %s", out_dir / "fit_split_half.png")


def _short_name(name: str) -> str:
    """Shorten parameter names for display."""
    replacements = {
        "sigma_halo_": "\u03c3h_",
        "sigma_shelf_": "\u03c3s_",
        "sigma_": "\u03c3_",
        "tau_halo_": "\u03c4h_",
        "tau_": "\u03c4_",
        "f_shelf_": "fsh_",
        "f_halo_": "fha_",
        "mu_": "\u03bc_",
        "sigma0": "\u03c30",
        "S_bkg": "S_bkg",
        "s_bkg": "S_bkg",
    }
    for pattern, replacement in replacements.items():
        if name.startswith(pattern) or name == pattern:
            return name.replace(pattern, replacement, 1)
    return name


# ═══════════════════════════════════════════════════════════════════════
#  Overfitting / goodness-of-fit diagnostic metrics
# ═══════════════════════════════════════════════════════════════════════


def compute_pull_diagnostics(
    fit_params: Mapping[str, object],
    min_counts: float = 5.0,
) -> dict:
    """Compute comprehensive pull-based overfitting diagnostics.

    Returns a dict of metrics suitable for inclusion in summary.json.

    Metrics
    -------
    pull_mean, pull_sigma : basic pull statistics
    pull_kurtosis : excess kurtosis (0 for Gaussian, >0 = heavy tails)
    pull_skewness : skewness (0 for Gaussian)
    frac_gt_2sigma_pct : % of pulls with |pull| > 2
    frac_gt_3sigma_pct : % of pulls with |pull| > 3
    durbin_watson : D ≈ 2 means no serial correlation; D < 2 = positive autocorr.
    autocorr_lag1 : lag-1 autocorrelation of pulls (ideal: 0)
    autocorr_lag2 : lag-2 autocorrelation
    autocorr_lag5 : lag-5 autocorrelation
    runs_test_z : z-score from the Wald-Wolfowitz runs test (|z| < 2 = OK)
    runs_observed : actual number of runs (sign changes + 1)
    runs_expected : expected runs under randomness
    ks_statistic, ks_pvalue : Kolmogorov-Smirnov test vs N(0,1)
    ad_statistic, ad_pvalue : Anderson-Darling test vs normal
    n_pulls : number of bins used
    """
    model = np.asarray(fit_params.get("_plot_model_total", []), dtype=float)
    hist = np.asarray(fit_params.get("_plot_hist", []), dtype=float)
    if model.size == 0 or hist.size == 0:
        return {}

    mask = model > min_counts
    pulls = (hist[mask] - model[mask]) / np.sqrt(model[mask])
    n = len(pulls)
    if n < 10:
        return {}

    result: dict = {}
    result["n_pulls"] = n

    # ── Basic statistics ──────────────────────────────────────────────
    result["pull_mean"] = round(float(np.mean(pulls)), 4)
    result["pull_sigma"] = round(float(np.std(pulls, ddof=1)), 4)
    result["pull_skewness"] = round(float(sp_stats.skew(pulls)), 4)
    result["pull_kurtosis"] = round(float(sp_stats.kurtosis(pulls)), 4)
    result["frac_gt_2sigma_pct"] = round(
        100.0 * np.sum(np.abs(pulls) > 2) / n, 2
    )
    result["frac_gt_3sigma_pct"] = round(
        100.0 * np.sum(np.abs(pulls) > 3) / n, 2
    )

    # ── Serial correlation ────────────────────────────────────────────
    # Durbin-Watson: D = sum(diff(pull)^2) / sum(pull^2)
    # D ≈ 2 → no autocorrelation; D < 2 → positive; D > 2 → negative
    ss = np.sum(pulls ** 2)
    if ss > 0:
        dw = float(np.sum(np.diff(pulls) ** 2) / ss)
    else:
        dw = 2.0
    result["durbin_watson"] = round(dw, 4)

    # Autocorrelation at specific lags
    pull_centered = pulls - np.mean(pulls)
    var = np.var(pulls)
    for lag in [1, 2, 5, 10]:
        if lag < n and var > 0:
            acf = float(np.mean(pull_centered[:-lag] * pull_centered[lag:])) / var
            result[f"autocorr_lag{lag}"] = round(acf, 4)

    # ── Wald-Wolfowitz runs test ──────────────────────────────────────
    # Counts runs of consecutive same-sign pulls; too few → systematic structure
    signs = np.sign(pulls)
    signs = signs[signs != 0]  # exclude exact zeros
    n_s = len(signs)
    if n_s > 1:
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
        result["runs_observed"] = runs
        if n_pos > 0 and n_neg > 0:
            expected = 1.0 + 2.0 * n_pos * n_neg / (n_pos + n_neg)
            var_runs = (
                2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_pos - n_neg)
                / ((n_pos + n_neg) ** 2 * (n_pos + n_neg - 1))
            )
            result["runs_expected"] = round(expected, 2)
            if var_runs > 0:
                z_runs = (runs - expected) / np.sqrt(var_runs)
                result["runs_test_z"] = round(float(z_runs), 4)

    # ── Normality tests ───────────────────────────────────────────────
    # KS test vs N(0,1)
    ks_stat, ks_p = sp_stats.kstest(pulls, "norm", args=(0, 1))
    result["ks_statistic"] = round(float(ks_stat), 6)
    result["ks_pvalue"] = float(ks_p)  # keep full precision

    # Anderson-Darling (tests normality, more sensitive in tails than KS)
    ad_result = sp_stats.anderson(pulls, dist="norm")
    result["ad_statistic"] = round(float(ad_result.statistic), 4)
    # Provide critical values and significance levels for interpretation
    result["ad_critical_values"] = {
        f"{sl}%": round(float(cv), 4)
        for sl, cv in zip(ad_result.significance_level, ad_result.critical_values)
    }

    # ── Rebinned diagnostics ──────────────────────────────────────────
    # Raw bin-level autocorrelation is dominated by ADC DNL (hardware
    # effect at ~0.013 MeV/bin scale).  Rebinning averages out DNL and
    # reveals true model-level structure.  We compute diagnostics at
    # two scales: ×10 (~0.13 MeV, peak-width scale) and ×30 (~0.4 MeV,
    # inter-peak scale).
    model_masked = model[mask]
    hist_masked = hist[mask]
    for rebin_factor in [5, 10, 30]:
        n_coarse = n // rebin_factor
        if n_coarse < 10:
            continue
        trim = n_coarse * rebin_factor
        m_r = model_masked[:trim].reshape(n_coarse, rebin_factor).sum(axis=1)
        h_r = hist_masked[:trim].reshape(n_coarse, rebin_factor).sum(axis=1)
        ok = m_r > rebin_factor * min_counts
        if ok.sum() < 10:
            continue
        p_r = (h_r[ok] - m_r[ok]) / np.sqrt(m_r[ok])
        nr = len(p_r)
        tag = f"rebin{rebin_factor}"
        result[f"{tag}_n_pulls"] = nr
        result[f"{tag}_pull_sigma"] = round(float(np.std(p_r, ddof=1)), 4)
        # Durbin-Watson
        ss_r = np.sum(p_r ** 2)
        if ss_r > 0:
            result[f"{tag}_durbin_watson"] = round(
                float(np.sum(np.diff(p_r) ** 2) / ss_r), 4
            )
        # Autocorrelation lag-1
        pc_r = p_r - np.mean(p_r)
        var_r = np.var(p_r)
        if var_r > 0 and nr > 1:
            result[f"{tag}_autocorr_lag1"] = round(
                float(np.mean(pc_r[:-1] * pc_r[1:]) / var_r), 4
            )
        # Runs test
        signs_r = np.sign(p_r)
        signs_r = signs_r[signs_r != 0]
        if len(signs_r) > 1:
            n_pos_r = int(np.sum(signs_r > 0))
            n_neg_r = int(np.sum(signs_r < 0))
            runs_r = 1 + int(np.sum(signs_r[1:] != signs_r[:-1]))
            result[f"{tag}_runs_observed"] = runs_r
            if n_pos_r > 0 and n_neg_r > 0:
                exp_r = 1.0 + 2.0 * n_pos_r * n_neg_r / (n_pos_r + n_neg_r)
                var_runs_r = (
                    2.0 * n_pos_r * n_neg_r
                    * (2.0 * n_pos_r * n_neg_r - n_pos_r - n_neg_r)
                    / ((n_pos_r + n_neg_r) ** 2 * (n_pos_r + n_neg_r - 1))
                )
                result[f"{tag}_runs_expected"] = round(exp_r, 2)
                if var_runs_r > 0:
                    result[f"{tag}_runs_test_z"] = round(
                        float((runs_r - exp_r) / np.sqrt(var_runs_r)), 4
                    )
        # KS test
        ks_r, ksp_r = sp_stats.kstest(p_r, "norm", args=(0, 1))
        result[f"{tag}_ks_pvalue"] = round(float(ksp_r), 6)

    return result


def plot_overfitting_diagnostics(
    fit_params: Mapping[str, object],
    pull_diagnostics: dict,
    out_dir: str | Path,
) -> None:
    """Create a 2×3 overfitting diagnostics figure.

    Panels:
      (a) Raw pull autocorrelation (bin-level, dominated by ADC DNL)
      (b) Rebinned pull autocorrelation (x10, model-level structure)
      (c) QQ plot of pulls vs N(0,1)
      (d) Pull run sequence (coloured by sign, vs energy)
      (e) Bin-level metric table (model adequacy)
      (f) Rebinned + overfitting metric table (overfitting checks)
    """
    out_dir = Path(out_dir)
    model = np.asarray(fit_params.get("_plot_model_total", []), dtype=float)
    hist = np.asarray(fit_params.get("_plot_hist", []), dtype=float)
    centers = np.asarray(fit_params.get("_plot_centers", []), dtype=float)
    if model.size == 0 or hist.size == 0:
        return

    mask = model > 5.0
    pulls = (hist[mask] - model[mask]) / np.sqrt(model[mask])
    energies = centers[mask] if centers.size == model.size else np.arange(len(pulls))
    n = len(pulls)
    if n < 10:
        return

    # Compute rebinned pulls for panel (b) - use x5 for more data points
    rebin = 5
    n_coarse = n // rebin
    trim = n_coarse * rebin
    m_r = model[mask][:trim].reshape(n_coarse, rebin).sum(axis=1)
    h_r = hist[mask][:trim].reshape(n_coarse, rebin).sum(axis=1)
    ok_r = m_r > rebin * 5.0
    pulls_rebinned = (h_r[ok_r] - m_r[ok_r]) / np.sqrt(m_r[ok_r])
    nr = len(pulls_rebinned)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # ── (a) Raw autocorrelation ───────────────────────────────────────
    ax = axes[0, 0]
    max_lag = min(30, n // 4)
    pc = pulls - np.mean(pulls)
    var = np.var(pulls)
    lags = np.arange(1, max_lag + 1)
    acf = np.array([
        np.mean(pc[:-lag] * pc[lag:]) / var if var > 0 else 0
        for lag in lags
    ])
    ax.bar(lags, acf, color="#4daf4a", alpha=0.7, edgecolor="white", linewidth=0.3)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, ls="--", color="red", alpha=0.5, lw=1)
    ax.axhline(-ci, ls="--", color="red", alpha=0.5, lw=1)
    ax.axhline(0, color="black", alpha=0.3, lw=0.5)
    ax.set_xlabel("Lag (bins, 0.013 MeV)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("(a) Raw ACF (bin-level, DNL dominated)", fontsize=9)
    ax.set_xlim(0.5, max_lag + 0.5)

    # ── (b) Rebinned autocorrelation ──────────────────────────────────
    ax = axes[0, 1]
    if nr > 10:
        max_lag_r = min(20, nr // 4)
        pc_r = pulls_rebinned - np.mean(pulls_rebinned)
        var_r = np.var(pulls_rebinned)
        lags_r = np.arange(1, max_lag_r + 1)
        acf_r = np.array([
            np.mean(pc_r[:-lag] * pc_r[lag:]) / var_r if var_r > 0 else 0
            for lag in lags_r
        ])
        ax.bar(lags_r, acf_r, color="#ff7f00", alpha=0.7, edgecolor="white",
               linewidth=0.3)
        ci_r = 1.96 / np.sqrt(nr)
        ax.axhline(ci_r, ls="--", color="red", alpha=0.5, lw=1)
        ax.axhline(-ci_r, ls="--", color="red", alpha=0.5, lw=1)
        ax.axhline(0, color="black", alpha=0.3, lw=0.5)
        ax.set_xlim(0.5, max_lag_r + 0.5)
    ax.set_xlabel(f"Lag (×{rebin} bins, ~{rebin*0.013:.2f} MeV)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"(b) Rebinned ×{rebin} ACF (model-level)", fontsize=9)

    # ── (c) QQ plot ───────────────────────────────────────────────────
    ax = axes[0, 2]
    sorted_pulls = np.sort(pulls)
    theoretical = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.scatter(theoretical, sorted_pulls, s=3, alpha=0.5, color="#377eb8")
    qq_lim = max(
        abs(theoretical.min()), abs(theoretical.max()),
        abs(sorted_pulls.min()), abs(sorted_pulls.max()),
    ) + 0.3
    ax.plot([-qq_lim, qq_lim], [-qq_lim, qq_lim], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("Theoretical quantiles [N(0,1)]")
    ax.set_ylabel("Pull quantiles")
    ax.set_title("(c) QQ Plot", fontsize=9)
    ax.set_xlim(-qq_lim, qq_lim)
    ax.set_ylim(-qq_lim, qq_lim)
    ax.set_aspect("equal")

    # ── (d) Pull run sequence ─────────────────────────────────────────
    ax = axes[1, 0]
    colors = np.where(pulls > 0, "#e41a1c", "#377eb8")
    bw = (energies[1] - energies[0]) * 0.9 if len(energies) > 1 else 0.01
    ax.bar(energies, pulls, width=bw, color=colors, alpha=0.5, edgecolor="none")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(2, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.axhline(-2, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("Pull (σ)")
    ax.set_title("(d) Pull Run Sequence", fontsize=9)

    # ── Helper: build and style a table ───────────────────────────────
    def _make_table(ax_t, rows, title, checks):
        ax_t.axis("off")
        col_labels = ["Metric", "Value", "Ideal", "Threshold"]
        table = ax_t.table(
            cellText=rows, colLabels=col_labels, loc="center",
            cellLoc="center", colWidths=[0.38, 0.20, 0.18, 0.24],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.45)
        for j in range(4):
            table[0, j].set_facecolor("#e2e3e5")
            table[0, j].set_text_props(fontweight="bold")
        for row_idx, ok in enumerate(checks, start=1):
            table[row_idx, 1].set_facecolor("#d4edda" if ok else "#f8d7da")
        ax_t.set_title(title, fontsize=9, pad=12)

    # ── (e) Bin-level model-adequacy table ────────────────────────────
    d = pull_diagnostics

    def _fv(key, fmt=".3f"):
        v = d.get(key, "?")
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)

    rows_e = [
        ["Pull σ (bin)", _fv("pull_sigma"), "1.00", "< 1.2"],
        ["|pull| > 2σ", _fv("frac_gt_2sigma_pct", ".1f") + "%", "4.6%", "< 10%"],
        ["|pull| > 3σ", _fv("frac_gt_3sigma_pct", ".1f") + "%", "0.3%", "< 2%"],
        ["Kurtosis", _fv("pull_kurtosis"), "0.00", "|κ| < 1"],
        ["DW (bin)", _fv("durbin_watson"), "2.00", "1.5-2.5"],
        ["ACF lag-1 (bin)", _fv("autocorr_lag1"), "0.00", "|r| < 0.1"],
        ["Runs z (bin)", _fv("runs_test_z", ".1f"), "0", "|z| < 3"],
        ["KS p-val (bin)", _fv("ks_pvalue", ".4f"), "> 0.05", "> 0.01"],
    ]
    checks_e = [
        d.get("pull_sigma", 9) < 1.2,
        d.get("frac_gt_2sigma_pct", 99) < 10,
        d.get("frac_gt_3sigma_pct", 99) < 2,
        abs(d.get("pull_kurtosis", 9)) < 1,
        1.5 < d.get("durbin_watson", 0) < 2.5,
        abs(d.get("autocorr_lag1", 1)) < 0.1,
        abs(d.get("runs_test_z", 99)) < 3,
        d.get("ks_pvalue", 0) > 0.01,
    ]
    _make_table(axes[1, 1], rows_e, "(e) Bin-level (model adequacy)", checks_e)

    # ── (f) Rebinned + overfitting table ──────────────────────────────
    def _safe_fmt(v, fmt=".3f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)

    def _n_label(tag):
        n_val = d.get(f"{tag}_n_pulls", "?")
        return f" (n={n_val})" if isinstance(n_val, int) else ""

    r5_sig = d.get("rebin5_pull_sigma", "?")
    r5_dw = d.get("rebin5_durbin_watson", "?")
    r5_ac = d.get("rebin5_autocorr_lag1", "?")
    r5_ks = d.get("rebin5_ks_pvalue", "?")
    r10_sig = d.get("rebin10_pull_sigma", "?")
    r10_dw = d.get("rebin10_durbin_watson", "?")
    r10_ac = d.get("rebin10_autocorr_lag1", "?")
    r10_rz = d.get("rebin10_runs_test_z", "?")
    r10_ks = d.get("rebin10_ks_pvalue", "?")

    rows_f = [
        [f"σ (×5){_n_label('rebin5')}", _safe_fmt(r5_sig), "1.00", "< 1.5"],
        [f"DW (×5)", _safe_fmt(r5_dw), "2.00", "1.5-2.5"],
        [f"ACF lag-1 (×5)", _safe_fmt(r5_ac), "0.00", "|r| < 0.15"],
        [f"KS p (×5)", _safe_fmt(r5_ks, ".4f"), "> 0.05", "> 0.01"],
        [f"σ (×10){_n_label('rebin10')}", _safe_fmt(r10_sig), "1.00", "< 1.5"],
        [f"DW (×10)", _safe_fmt(r10_dw), "2.00", "1.5-2.5"],
        [f"ACF lag-1 (×10)", _safe_fmt(r10_ac), "0.00", "|r| < 0.15"],
        [f"Runs z (×10)", _safe_fmt(r10_rz, ".1f"), "0", "|z| < 3"],
    ]
    checks_f = [
        (r5_sig < 1.5) if isinstance(r5_sig, (int, float)) else False,
        (1.5 < r5_dw < 2.5) if isinstance(r5_dw, (int, float)) else False,
        (abs(r5_ac) < 0.15) if isinstance(r5_ac, (int, float)) else False,
        (r5_ks > 0.01) if isinstance(r5_ks, (int, float)) else False,
        (r10_sig < 1.5) if isinstance(r10_sig, (int, float)) else False,
        (1.5 < r10_dw < 2.5) if isinstance(r10_dw, (int, float)) else False,
        (abs(r10_ac) < 0.15) if isinstance(r10_ac, (int, float)) else False,
        (abs(r10_rz) < 3) if isinstance(r10_rz, (int, float)) else False,
    ]
    _make_table(axes[1, 2], rows_f, "(f) Rebinned (overfitting check)", checks_f)

    fig.suptitle("Overfitting & Model-Adequacy Diagnostics",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fit_overfitting_diagnostics.png", dpi=200)
    plt.close(fig)
    logger.info(
        "Saved overfitting diagnostics to %s",
        out_dir / "fit_overfitting_diagnostics.png",
    )


def plot_model_comparison(
    model_comparison: dict,
    out_dir: str | Path,
) -> None:
    """Plot AIC/BIC comparison of full vs reduced models.

    Shows a grouped bar chart of ΔAIC and ΔBIC for each reduced model.
    Positive values mean the full model is preferred; negative means
    the simpler model is preferred.
    """
    out_dir = Path(out_dir)
    base = model_comparison.get("base_model", {})
    reduced = model_comparison.get("reduced_models", [])
    if not reduced:
        return

    labels = []
    d_aic = []
    d_bic = []
    d_chi2 = []
    n_free = []
    for rm in reduced:
        labels.append(rm["description"])
        d_aic.append(rm["delta_aic"])
        d_bic.append(rm["delta_bic"])
        d_chi2.append(rm["delta_chi2"])
        n_free.append(rm["n_free"])

    n = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, 2.0 + 0.8 * n)))

    # Panel (a): ΔAIC and ΔBIC bars
    y = np.arange(n)
    bar_h = 0.35
    ax = axes[0]
    bars_aic = ax.barh(y + bar_h / 2, d_aic, bar_h, label="ΔAIC", color="#4C72B0")
    bars_bic = ax.barh(y - bar_h / 2, d_bic, bar_h, label="ΔBIC", color="#DD8452")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δ Information Criterion (vs Full Model)")
    ax.set_title("(a) Model Selection: ΔAIC / ΔBIC", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    # Annotate: positive = full preferred, negative = simpler preferred
    xlim = ax.get_xlim()
    if xlim[1] > 0:
        ax.text(
            xlim[1] * 0.7, n - 0.5,
            "← simpler preferred   full preferred →",
            fontsize=7, ha="center", va="top", style="italic", color="gray",
        )
    ax.invert_yaxis()

    # Panel (b): Summary table
    ax2 = axes[1]
    ax2.axis("off")
    col_labels = ["Model", "k", "Δχ²", "ΔAIC", "ΔBIC", "Verdict"]
    table_data = []
    cell_colors = []
    # Base model row
    table_data.append([
        "Full (current)",
        str(base.get("n_free", "?")),
        "-",
        "-",
        "-",
        "baseline",
    ])
    cell_colors.append(["#f0f0f0"] * 6)
    for i, rm in enumerate(reduced):
        verdict = "full preferred" if d_bic[i] > 2 else (
            "simpler preferred" if d_bic[i] < -2 else "comparable"
        )
        color = (
            "#d4edda" if d_bic[i] > 2 else
            "#f8d7da" if d_bic[i] < -2 else
            "#fff3cd"
        )
        table_data.append([
            labels[i],
            str(n_free[i]),
            f"{d_chi2[i]:+.1f}",
            f"{d_aic[i]:+.1f}",
            f"{d_bic[i]:+.1f}",
            verdict,
        ])
        cell_colors.append([color] * 6)

    tbl = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.5)
    # Bold header
    for j in range(len(col_labels)):
        tbl[0, j].set_text_props(fontweight="bold")
    ax2.set_title(
        "(b) Model Comparison Summary\n"
        f"(BIC interpretation: ΔBIC > +2 = full preferred,  "
        f"ΔBIC < −2 = simpler preferred)",
        fontsize=9, fontweight="bold",
    )

    conclusion = model_comparison.get("conclusion", "unknown")
    fig.suptitle(
        f"Model Complexity Comparison - {conclusion.replace('_', ' ').title()}",
        fontsize=12, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "fit_model_comparison.png", dpi=200)
    plt.close(fig)
    logger.info(
        "Saved model comparison to %s",
        out_dir / "fit_model_comparison.png",
    )


# ═══════════════════════════════════════════════════════════════════════
# Extended residual diagnostics: multi-type residuals, regional RMS,
# power spectrum
# ═══════════════════════════════════════════════════════════════════════


def _pearson_residuals(
    hist: np.ndarray, model: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """Pearson residual: (d_i - m_i) / sqrt(m_i)."""
    return (hist[mask] - model[mask]) / np.sqrt(model[mask])


def _deviance_residuals(
    hist: np.ndarray, model: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """Deviance residual for Poisson data.

    r_i^(D) = sign(d - m) * sqrt(2 * [d*ln(d/m) - (d - m)])
    Handles d=0 gracefully (term reduces to 2*m).
    """
    d = hist[mask].astype(float)
    m = model[mask].astype(float)
    sign = np.sign(d - m)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(d > 0, d * np.log(d / m), 0.0) - (d - m)
    # Clamp numerical noise below zero
    term = np.maximum(term, 0.0)
    return sign * np.sqrt(2.0 * term)


def _randomized_quantile_residuals(
    hist: np.ndarray, model: np.ndarray, mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomized quantile residuals for Poisson data.

    For each bin with observed count d_i and expected m_i:
      F(d_i; m_i) = Poisson CDF at d_i
      F(d_i - 1; m_i) = Poisson CDF at d_i - 1
      u ~ Uniform(F(d_i-1), F(d_i))
      r_i = Phi^{-1}(u)
    These should be exactly N(0,1) if the model is correct.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    d = hist[mask].astype(float)
    m = model[mask].astype(float)
    cdf_upper = sp_stats.poisson.cdf(d, m)
    cdf_lower = sp_stats.poisson.cdf(d - 1, m)
    # Clamp to avoid exactly 0 or 1 (would give ±inf from ppf)
    eps = 1e-12
    cdf_lower = np.clip(cdf_lower, eps, 1.0 - eps)
    cdf_upper = np.clip(cdf_upper, eps, 1.0 - eps)
    u = rng.uniform(cdf_lower, cdf_upper)
    return sp_stats.norm.ppf(u)


def compute_extended_residual_diagnostics(
    fit_params: Mapping[str, object],
    min_counts: float = 5.0,
) -> dict:
    """Compute Pearson, deviance, and randomized quantile residuals.

    Returns a dict with per-type statistics, regional RMS, and power
    spectrum data suitable for plotting and inclusion in summary.json.
    """
    model = np.asarray(fit_params.get("_plot_model_total", []), dtype=float)
    hist = np.asarray(fit_params.get("_plot_hist", []), dtype=float)
    centers = np.asarray(fit_params.get("_plot_centers", []), dtype=float)
    if model.size == 0 or hist.size == 0 or centers.size == 0:
        return {}

    mask = model > min_counts
    n = int(mask.sum())
    if n < 10:
        return {}

    energies = centers[mask]
    result: dict = {"n_bins": n}

    # ── Compute all three residual types ──────────────────────────────
    pearson = _pearson_residuals(hist, model, mask)
    deviance = _deviance_residuals(hist, model, mask)
    rqr = _randomized_quantile_residuals(hist, model, mask)

    residual_types = {
        "pearson": pearson,
        "deviance": deviance,
        "rqr": rqr,
    }

    # ── Per-type summary statistics ───────────────────────────────────
    for rtype, resid in residual_types.items():
        finite = resid[np.isfinite(resid)]
        if len(finite) < 5:
            continue
        result[f"{rtype}_mean"] = round(float(np.mean(finite)), 4)
        result[f"{rtype}_rms"] = round(float(np.sqrt(np.mean(finite ** 2))), 4)
        result[f"{rtype}_sigma"] = round(float(np.std(finite, ddof=1)), 4)
        result[f"{rtype}_skewness"] = round(float(sp_stats.skew(finite)), 4)
        result[f"{rtype}_kurtosis"] = round(float(sp_stats.kurtosis(finite)), 4)
        ks_stat, ks_p = sp_stats.kstest(finite, "norm", args=(0, 1))
        result[f"{rtype}_ks_pvalue"] = round(float(ks_p), 6)

    # ── Regional RMS (by isotope peak and inter-peak gaps) ────────────
    all_regions = {**_PEAK_REGIONS, **_INTER_PEAK_REGIONS}
    for region_name, (lo, hi) in all_regions.items():
        rmask = (energies >= lo) & (energies <= hi)
        n_r = int(rmask.sum())
        if n_r < 3:
            continue
        for rtype, resid in residual_types.items():
            r_slice = resid[rmask]
            finite = r_slice[np.isfinite(r_slice)]
            if len(finite) < 3:
                continue
            rms = float(np.sqrt(np.mean(finite ** 2)))
            result[f"region_{region_name}_{rtype}_rms"] = round(rms, 4)
        result[f"region_{region_name}_n_bins"] = n_r

    # ── Power spectrum of Pearson residuals (FFT) ─────────────────────
    # Use Pearson as the primary; if user wants others, the arrays are
    # stored for the plot function.
    finite_pearson = pearson[np.isfinite(pearson)]
    if len(finite_pearson) >= 16:
        # Subtract mean to remove DC component, then FFT
        centered = finite_pearson - np.mean(finite_pearson)
        N = len(centered)
        fft_vals = sp_fft.rfft(centered)
        power = np.abs(fft_vals) ** 2 / N
        freqs = sp_fft.rfftfreq(N)  # in cycles per bin
        # Store only summary stats (full arrays go to plot function)
        # Exclude DC (index 0)
        if len(power) > 1:
            power_no_dc = power[1:]
            freqs_no_dc = freqs[1:]
            # Mean power should be ~1.0 for white noise (Pearson N(0,1))
            result["fft_mean_power"] = round(float(np.mean(power_no_dc)), 4)
            result["fft_max_power"] = round(float(np.max(power_no_dc)), 4)
            result["fft_max_freq"] = round(float(freqs_no_dc[np.argmax(power_no_dc)]), 6)
            # Ratio of max to mean - large values indicate spectral peaks
            if np.mean(power_no_dc) > 0:
                result["fft_peak_ratio"] = round(
                    float(np.max(power_no_dc) / np.mean(power_no_dc)), 2
                )

    # ── Stash arrays for the plotting function (not serialized) ───────
    result["_arrays"] = {
        "energies": energies,
        "pearson": pearson,
        "deviance": deviance,
        "rqr": rqr,
    }

    return result


def plot_extended_residual_diagnostics(
    fit_params: Mapping[str, object],
    ext_diagnostics: dict,
    out_dir: str | Path,
) -> None:
    """Create extended residual diagnostics figure.

    Panels (3×2):
      (a) Pearson vs Deviance vs RQR residuals overlaid
      (b) QQ plots for all three residual types
      (c) Regional RMS bar chart
      (d) Power spectrum of Pearson residuals
      (e) Power spectrum of deviance residuals
      (f) Residual comparison summary table
    """
    out_dir = Path(out_dir)
    arrays = ext_diagnostics.get("_arrays")
    if arrays is None:
        return

    energies = arrays["energies"]
    pearson = arrays["pearson"]
    deviance = arrays["deviance"]
    rqr = arrays["rqr"]
    n = len(pearson)
    if n < 10:
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # ── (a) Three residual types vs energy ────────────────────────────
    ax = axes[0, 0]
    ax.scatter(energies, pearson, s=1.5, alpha=0.4, color="#1f77b4", label="Pearson")
    ax.scatter(energies, deviance, s=1.5, alpha=0.4, color="#ff7f0e", label="Deviance")
    ax.scatter(energies, rqr, s=1.5, alpha=0.4, color="#2ca02c", label="RQR")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(2, ls="--", color="gray", alpha=0.5, lw=0.7)
    ax.axhline(-2, ls="--", color="gray", alpha=0.5, lw=0.7)
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("Residual (σ)")
    ax.set_title("(a) Residual Comparison vs Energy", fontsize=9)
    ax.legend(fontsize=7, markerscale=4, loc="upper right")
    ax.set_ylim(-5, 5)

    # ── (b) QQ plots for all three types ──────────────────────────────
    ax = axes[0, 1]
    theoretical = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    for resid, color, label in [
        (pearson, "#1f77b4", "Pearson"),
        (deviance, "#ff7f0e", "Deviance"),
        (rqr, "#2ca02c", "RQR"),
    ]:
        finite = resid[np.isfinite(resid)]
        if len(finite) < 5:
            continue
        sorted_r = np.sort(finite)
        # Handle size mismatch if some were filtered
        th = sp_stats.norm.ppf((np.arange(1, len(sorted_r) + 1) - 0.5) / len(sorted_r))
        ax.scatter(th, sorted_r, s=2, alpha=0.4, color=color, label=label)
    qq_lim = 4.5
    ax.plot([-qq_lim, qq_lim], [-qq_lim, qq_lim], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("Theoretical quantiles [N(0,1)]")
    ax.set_ylabel("Residual quantiles")
    ax.set_title("(b) QQ Plots: Pearson / Deviance / RQR", fontsize=9)
    ax.set_xlim(-qq_lim, qq_lim)
    ax.set_ylim(-qq_lim, qq_lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, markerscale=4)

    # ── (c) Regional RMS bar chart ────────────────────────────────────
    ax = axes[1, 0]
    all_regions = list(_PEAK_REGIONS.keys()) + list(_INTER_PEAK_REGIONS.keys())
    rms_data = {}
    for rtype in ["pearson", "deviance", "rqr"]:
        vals = []
        for region in all_regions:
            key = f"region_{region}_{rtype}_rms"
            vals.append(ext_diagnostics.get(key, np.nan))
        rms_data[rtype] = np.array(vals)

    x = np.arange(len(all_regions))
    bar_w = 0.25
    for i, (rtype, color) in enumerate([
        ("pearson", "#1f77b4"), ("deviance", "#ff7f0e"), ("rqr", "#2ca02c"),
    ]):
        ax.bar(x + i * bar_w, rms_data[rtype], bar_w, color=color, alpha=0.8,
               label=rtype.capitalize())
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5, label="Ideal (1.0)")
    ax.set_xticks(x + bar_w)
    short_labels = [r.replace("gap_", "").replace("_", "-") for r in all_regions]
    ax.set_xticklabels(short_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("RMS")
    ax.set_title("(c) Regional RMS by Residual Type", fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    # ── (d) Power spectrum of Pearson residuals ───────────────────────
    ax = axes[1, 1]
    _plot_power_spectrum(ax, pearson, "(d) Power Spectrum - Pearson", "#1f77b4")

    # ── (e) Power spectrum of deviance residuals ──────────────────────
    ax = axes[2, 0]
    _plot_power_spectrum(ax, deviance, "(e) Power Spectrum - Deviance", "#ff7f0e")

    # ── (f) Summary comparison table ──────────────────────────────────
    ax = axes[2, 1]
    ax.axis("off")
    col_labels = ["Metric", "Pearson", "Deviance", "RQR"]
    rows = []
    for metric, fmt in [
        ("rms", ".3f"), ("sigma", ".3f"), ("mean", ".3f"),
        ("skewness", ".3f"), ("kurtosis", ".3f"), ("ks_pvalue", ".4f"),
    ]:
        row = [metric.replace("_", " ").title()]
        for rtype in ["pearson", "deviance", "rqr"]:
            v = ext_diagnostics.get(f"{rtype}_{metric}", "?")
            row.append(f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v))
        rows.append(row)

    # Add FFT summary row
    fft_peak = ext_diagnostics.get("fft_peak_ratio", "?")
    rows.append([
        "FFT Peak Ratio",
        f"{fft_peak:.1f}" if isinstance(fft_peak, (int, float)) else str(fft_peak),
        "-", "-",
    ])

    table = ax.table(
        cellText=rows, colLabels=col_labels, loc="center",
        cellLoc="center", colWidths=[0.28, 0.24, 0.24, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for j in range(4):
        table[0, j].set_facecolor("#e2e3e5")
        table[0, j].set_text_props(fontweight="bold")
    # Highlight: green if RMS close to 1, red if far
    for row_idx in range(1, len(rows) + 1):
        if rows[row_idx - 1][0] == "Rms":
            for col_idx in range(1, 4):
                try:
                    v = float(rows[row_idx - 1][col_idx])
                    ok = 0.7 < v < 1.3
                    table[row_idx, col_idx].set_facecolor(
                        "#d4edda" if ok else "#f8d7da"
                    )
                except (ValueError, IndexError):
                    pass
    ax.set_title("(f) Residual Type Comparison", fontsize=9, pad=12)

    fig.suptitle(
        "Extended Residual Diagnostics: Pearson / Deviance / RQR",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "fit_extended_residuals.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "Saved extended residual diagnostics to %s",
        out_dir / "fit_extended_residuals.png",
    )


def _plot_power_spectrum(
    ax: plt.Axes,
    residuals: np.ndarray,
    title: str,
    color: str,
) -> None:
    """Plot the power spectrum of a residual array on log-log axes.

    Log frequency axis spreads out the low-frequency region where model
    misfit and DNL structure live, instead of wasting space on flat
    high-frequency noise. Secondary top axis shows period in bins.
    """
    finite = residuals[np.isfinite(residuals)]
    if len(finite) < 16:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title(title, fontsize=9)
        return

    centered = finite - np.mean(finite)
    N = len(centered)
    fft_vals = sp_fft.rfft(centered)
    power = np.abs(fft_vals) ** 2 / N
    freqs = sp_fft.rfftfreq(N)

    # Skip DC component
    if len(power) > 1:
        freqs = freqs[1:]
        power = power[1:]

    # White noise expectation: mean power ~ variance ~ 1 for N(0,1) residuals
    expected_power = np.var(finite)

    # Log-log plot
    ax.loglog(freqs, power, color=color, alpha=0.4, lw=0.5)

    # Smoothed version using log-spaced binning (geometric averaging)
    if len(power) > 20:
        n_smooth_bins = min(40, len(power) // 3)
        log_edges = np.geomspace(freqs[0], freqs[-1], n_smooth_bins + 1)
        smooth_f = []
        smooth_p = []
        for i in range(n_smooth_bins):
            mask = (freqs >= log_edges[i]) & (freqs < log_edges[i + 1])
            if np.any(mask):
                smooth_f.append(np.sqrt(log_edges[i] * log_edges[i + 1]))
                smooth_p.append(np.mean(power[mask]))
        if smooth_f:
            ax.loglog(smooth_f, smooth_p, color=color, lw=2.0, alpha=0.9,
                      label="Smoothed")

    ax.axhline(expected_power, color="k", ls="--", lw=0.8, alpha=0.6,
               label=f"White noise ({expected_power:.2f})")
    # 95% confidence for white noise (chi2 with 2 DOF per frequency bin)
    ci_95 = expected_power * sp_stats.chi2.ppf(0.95, 2) / 2
    ax.axhline(ci_95, color="red", ls=":", lw=0.7, alpha=0.5,
               label=f"95% CL ({ci_95:.2f})")
    ax.set_ylabel("Power")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)

    # Frequency axis: log scale, from lowest resolvable to Nyquist
    f_min = freqs[0] * 0.8 if len(freqs) > 0 else 1e-3
    ax.set_xlim(f_min, 0.5)
    ax.set_xlabel("Frequency [cycles/bin]")

    # Secondary top axis: period in bins
    ax2 = ax.secondary_xaxis(
        "top",
        functions=(lambda f: np.where(f > 0, 1.0 / f, np.inf),
                   lambda p: np.where(p > 0, 1.0 / p, np.inf)),
    )
    ax2.set_xlabel("Period [bins]", fontsize=7)
    # Pick period ticks that fall within the frequency range
    all_ticks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    period_max = 1.0 / f_min if f_min > 0 else 1000
    valid_ticks = [t for t in all_ticks if 2 <= t <= period_max]
    ax2.set_ticks(valid_ticks)
    ax2.set_xticklabels([str(t) for t in valid_ticks], fontsize=6)


# ---------------------------------------------------------------------------
# Lucas-cell assay bridge summary plot
# ---------------------------------------------------------------------------

def plot_bridge_summary(
    bridge_results: dict,
    out_dir: str | Path,
    spike_results: dict | None = None,
) -> None:
    """3-panel diagnostic plot for the Lucas-cell assay bridge.

    Panel 1: Bridge factor per assay (scatter + error bars + aggregate band)
    Panel 2: Detection efficiency per assay (scatter + error bars + aggregate band)
              + spike-derived efficiency predictions if available
    Panel 3: Reference vs measured activity (log-scale grouped bars)

    Parameters
    ----------
    bridge_results : dict
        The ``lucas_bridge`` section from the pipeline summary.
    out_dir : str or Path
        Directory for the output PNG.
    spike_results : dict, optional
        Output from ``fit_spike_periods()`` - spike-derived efficiency
        predictions are overlaid on Panel 2.
    """
    out_dir = Path(out_dir)
    assays = bridge_results.get("assays", [])
    if not assays:
        logger.info("No assay data for bridge summary plot; skipping")
        return

    from datetime import datetime

    # --- Parse per-assay data ---
    dates: list[datetime] = []
    labels: list[str] = []
    bfs: list[float] = []
    bf_uncs: list[float] = []
    effs: list[float] = []
    eff_uncs: list[float] = []
    ref_bqs: list[float] = []
    ref_uncs: list[float] = []
    meas_bqs: list[float] = []
    meas_uncs: list[float] = []
    categories: list[str] = []  # "RM", "UI", "BG"
    in_agg: list[bool] = []

    for a in assays:
        date_str = a.get("assay_date")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str)
        except (ValueError, TypeError):
            continue

        bf = a.get("bridge_factor")
        bf_u = a.get("bridge_factor_unc")
        if bf is None or bf_u is None:
            continue

        dates.append(dt)
        labels.append(a.get("assay_label", "")[:30])
        bfs.append(bf)
        bf_uncs.append(bf_u)
        effs.append(a.get("detection_efficiency") or 0)
        eff_uncs.append(a.get("detection_efficiency_unc") or 0)
        ref_bqs.append(a.get("reference_activity_bq") or 0)
        ref_uncs.append(a.get("reference_activity_unc_bq") or 0)
        meas_bqs.append(a.get("measured_activity_bq") or 0)
        meas_uncs.append(a.get("measured_activity_unc_bq") or 0)

        in_agg.append(a.get("in_aggregate", True))

        lbl = a.get("assay_label", "").lower()
        is_bg = a.get("is_background_assay", False) or "background" in lbl
        if is_bg:
            categories.append("BG")
        elif "ui connected" in lbl or "ui through" in lbl:
            categories.append("UI")
        else:
            categories.append("RM")

    if not dates:
        logger.info("No valid assay data for bridge plot; skipping")
        return

    # Color map
    cat_colors = {"RM": "#1f77b4", "UI": "#ff7f0e", "BG": "#7f7f7f"}
    colors = [cat_colors.get(c, "#1f77b4") for c in categories]

    # Aggregate values
    agg = bridge_results.get("aggregate", {})
    agg_bf = agg.get("weighted_mean_bridge_factor")
    agg_bf_unc = agg.get("weighted_mean_unc")

    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    fig.suptitle("Lucas-Cell Assay Bridge Summary", fontsize=13, fontweight="bold")

    # ── Panel 1: Bridge Factor ──
    ax1 = axes[0]
    from matplotlib.lines import Line2D
    _MS = 3          # marker size (small so error bars visible)
    _MEW = 0.6       # marker edge width
    _ELW = 0.9       # error bar line width
    _ALPHA = 0.75    # marker translucency
    for i, (d, bf, bfu, col, ia) in enumerate(zip(dates, bfs, bf_uncs, colors, in_agg)):
        mfc = col if ia else "none"
        ax1.errorbar(d, bf, yerr=bfu, fmt="o", color=col, capsize=3,
                     markersize=_MS, elinewidth=_ELW, zorder=5,
                     markerfacecolor=mfc, markeredgecolor=col,
                     markeredgewidth=_MEW, alpha=_ALPHA)
    if agg_bf is not None and agg_bf_unc is not None:
        ax1.axhspan(agg_bf - agg_bf_unc, agg_bf + agg_bf_unc,
                     alpha=0.15, color="#2ca02c", zorder=1)
        ax1.axhline(agg_bf, color="#2ca02c", ls="--", lw=1, alpha=0.7,
                     label=f"Weighted mean = {agg_bf:.2f} +/- {agg_bf_unc:.2f}")
    ax1.set_ylabel("Bridge Factor (ref / meas)")
    ax1.set_title("Bridge Factor per Assay", fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.tick_params(axis="x", rotation=30, labelsize=7)
    # Category legend
    legend_handles = []
    for cat, col in cat_colors.items():
        if cat in categories:
            legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                         markerfacecolor=col, markersize=3.5, label=cat))
    if agg_bf is not None:
        legend_handles.append(Line2D([0], [0], ls="--", color="#2ca02c",
                                      label=f"Weighted mean = {agg_bf:.2f} ± {agg_bf_unc:.2f}"))
    ax1.legend(handles=legend_handles, fontsize=7, loc="upper right")

    # ── Panel 2: Detection Efficiency ──
    ax2 = axes[1]
    for i, (d, eff, effu, col, ia) in enumerate(zip(dates, effs, eff_uncs, colors, in_agg)):
        mfc = col if ia else "none"
        # Clip display values to [0, 1] but keep error bars proportional
        eff_disp = min(eff, 1.0)
        # Show error bar even if point is at edge
        yerr_lo = min(effu, eff_disp) if effu > 0 else 0
        yerr_hi = effu
        ax2.errorbar(d, eff_disp, yerr=[[yerr_lo], [yerr_hi]], fmt="o",
                     color=col, capsize=3, markersize=_MS, elinewidth=_ELW,
                     zorder=5, markerfacecolor=mfc, markeredgecolor=col,
                     markeredgewidth=_MEW, alpha=_ALPHA)
        # Annotate relative uncertainty
        if eff > 0 and effu > 0:
            rel_pct = effu / eff * 100
            y_ann = min(eff_disp + effu + 0.02, 0.95)
            ax2.annotate(f"±{rel_pct:.0f}%", (d, y_ann), fontsize=5.5,
                         ha="center", color=col, alpha=0.7)
    # Aggregate efficiency = 1/BF
    if agg_bf is not None and agg_bf > 0 and agg_bf_unc is not None:
        agg_eff = 1.0 / agg_bf
        agg_eff_unc = agg_bf_unc / agg_bf**2
        ax2.axhspan(max(agg_eff - agg_eff_unc, 0), min(agg_eff + agg_eff_unc, 1),
                     alpha=0.15, color="#2ca02c", zorder=1)
        ax2.axhline(agg_eff, color="#2ca02c", ls="--", lw=1, alpha=0.7,
                     label=f"Aggregate = {agg_eff:.3f} ± {agg_eff_unc:.3f}")
    # Overlay spike-derived efficiency on Panel 2
    if spike_results is not None:
        spike_periods = spike_results.get("periods", [])
        for sp in spike_periods:
            if sp.get("error"):
                continue
            # Use spike_efficiency = R0 / ref_bq when available
            spike_eff = sp.get("spike_efficiency")
            spike_eff_unc = sp.get("spike_efficiency_unc", 0)
            if spike_eff is None or spike_eff <= 0:
                continue
            label_sp = sp.get("label", sp.get("t_start", ""))
            # Place the spike marker at the matched assay date, not the
            # spike injection start date.
            date_str = sp.get("matched_assay_date") or sp.get("t_start")
            try:
                sp_date = datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                continue
            ax2.errorbar(sp_date, spike_eff, yerr=spike_eff_unc,
                         fmt="d", color="#d62728",
                         capsize=2, markersize=3, elinewidth=0.8, zorder=10,
                         markerfacecolor="#d62728", markeredgecolor="k",
                         markeredgewidth=0.4, alpha=0.75,
                         label=f"Spike eff: {label_sp}")
        if any(sp.get("spike_efficiency") for sp in spike_periods):
            ax2.annotate("◆ = Spike-derived efficiency (R₀/ref)",
                         xy=(0.02, 0.02), xycoords="axes fraction",
                         fontsize=5.5, color="#d62728", alpha=0.7)

    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Detection Efficiency (1/BF)")
    ax2.set_title("Detection Efficiency per Assay", fontsize=10)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.tick_params(axis="x", rotation=30, labelsize=7)

    # ── Panel 3: Reference vs Measured Activity (log scale) ──
    ax3 = axes[2]
    n = len(dates)
    x = np.arange(n)
    bar_w = 0.35
    bars_ref = ax3.bar(x - bar_w / 2, ref_bqs, bar_w, yerr=ref_uncs,
                       label="Reference (assay)", color="#1f77b4", alpha=0.7,
                       capsize=2, ecolor="#333")
    bars_meas = ax3.bar(x + bar_w / 2, meas_bqs, bar_w, yerr=meas_uncs,
                        label="Measured (monitor)", color="#ff7f0e", alpha=0.7,
                        capsize=2, ecolor="#333")
    ax3.set_yscale("log")
    ax3.set_xticks(x)
    short_labels = [f"{d.strftime('%m/%d')}\n{l[:20]}" for d, l in zip(dates, labels)]
    ax3.set_xticklabels(short_labels, fontsize=6, rotation=30, ha="right")
    ax3.set_ylabel("Activity (Bq)")
    ax3.set_title("Reference vs Measured Activity", fontsize=10)
    ax3.legend(fontsize=8)

    # Save
    out_path = out_dir / "assay_bridge_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved assay bridge summary to %s", out_path)


def plot_spike_decay_fits(
    spike_results: dict,
    isotope_series: dict,
    out_dir: str | Path,
) -> None:
    """Plot spike decay fits - data + model for each spike period.

    Creates one subplot per spike period showing:
    - Binned count rate data (isotopes summed)
    - Best-fit exponential decay curve
    - Residuals

    Parameters
    ----------
    spike_results : dict
        Output from ``fit_spike_periods()``.
    isotope_series : dict
        Per-isotope time series data.
    out_dir : str or Path
        Output directory for the plot.
    """
    import math
    from datetime import datetime

    out_dir = Path(out_dir)
    periods = spike_results.get("periods", [])
    valid_periods = [p for p in periods if not p.get("error") and p.get("R0")]

    if not valid_periods:
        logger.info("No valid spike fits to plot")
        return

    n = len(valid_periods)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4.5 * n), squeeze=False)

    isotopes = ("Po214", "Po218")
    half_life_days = spike_results.get("half_life_days_fixed", 3.8235)
    lam = math.log(2) / (half_life_days * 86400.0)

    for idx, period in enumerate(valid_periods):
        ax = axes[idx, 0]

        t_start_iso = period["t_start"]
        t_end_iso = period["t_end"]
        t0 = datetime.fromisoformat(t_start_iso).timestamp()
        t1 = datetime.fromisoformat(t_end_iso).timestamp()

        R0 = period["R0"]
        B = period["B"]
        R0_unc = period.get("R0_unc", 0)
        B_unc = period.get("B_unc", 0)
        chi2_ndf = period.get("chi2_ndf", float("nan"))
        label = period.get("label", f"{t_start_iso} → {t_end_iso}")

        # Collect binned data
        time_bins: dict[float, list[tuple[float, float]]] = {}
        for iso in isotopes:
            for entry in isotope_series.get(iso, []):
                t = entry.get("t")
                c = entry.get("counts", 0.0)
                dt = entry.get("dt", 0.0)
                if t is None or not np.isfinite(t) or dt <= 0:
                    continue
                if t0 <= t <= t1:
                    if t not in time_bins:
                        time_bins[t] = []
                    time_bins[t].append((c, dt))

        if not time_bins:
            continue

        t_arr = []
        r_arr = []
        u_arr = []
        for t_mid in sorted(time_bins.keys()):
            entries = time_bins[t_mid]
            total_c = sum(c for c, _ in entries)
            dt = entries[0][1]
            t_arr.append(t_mid)
            r_arr.append(total_c / dt)
            u_arr.append(math.sqrt(max(total_c, 1.0)) / dt)

        t_arr = np.array(t_arr)
        r_arr = np.array(r_arr)
        u_arr = np.array(u_arr)

        # Time in days since start
        t_days = (t_arr - t0) / 86400.0
        skip_days = period.get("skip_initial_days", 0.5)

        # Split data into skipped (injection) and fitted regions
        fit_mask = t_days >= skip_days
        skip_mask = ~fit_mask

        # Model curve (smooth) - starts from skip_days
        t_model = np.linspace(skip_days, (t1 - t0) / 86400.0, 200)
        r_model = R0 * np.exp(-lam * t_model * 86400.0) + B

        # Plot skipped bins (grayed out)
        if np.any(skip_mask):
            ax.errorbar(t_days[skip_mask], r_arr[skip_mask], yerr=u_arr[skip_mask],
                         fmt=".", color="gray", markersize=3, alpha=0.3,
                         label=f"Skipped ({skip_days:.1f}d injection)")
        # Plot fitted bins
        ax.errorbar(t_days[fit_mask], r_arr[fit_mask], yerr=u_arr[fit_mask],
                     fmt=".", color="C0", markersize=3, alpha=0.6,
                     label="Data (Po214+Po218)")
        ax.plot(t_model, r_model, "-", color="C3", lw=1.5,
                label=(f"Fit: R₀={R0:.4f}±{R0_unc:.4f}, "
                       f"B={B:.5f}±{B_unc:.5f}\n"
                       f"χ²/ndf={chi2_ndf:.2f}"))
        ax.axhline(B, color="gray", ls="--", alpha=0.5, label=f"Baseline B={B:.5f}")
        # Mark the skip boundary
        ax.axvline(skip_days, color="orange", ls=":", alpha=0.6, lw=1)

        ax.set_xlabel(f"Days since {t_start_iso}")
        ax.set_ylabel("Count rate (Hz)")
        ax.set_title(f"{label}  (t½={half_life_days:.4f} d fixed)")
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out_path = out_dir / "spike_decay_fits.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved spike decay fits to %s", out_path)


def plot_dnl_crossval(
    crossval_result,
    out_dir: str | Path,
    bin_centers: np.ndarray | None = None,
) -> None:
    """Three-panel DNL cross-validation diagnostic plot.

    Panel 1: DNL factors from half-A vs half-B (scatter + correlation).
    Panel 2: NLL comparison bar chart (no-DNL / self-DNL / cross-DNL).
    Panel 3: Per-bin signed delta-NLL contribution from cross-DNL vs no-DNL.

    Parameters
    ----------
    crossval_result : DNLCrossValResult
        Result from ``run_dnl_crossval``.
    out_dir : str or Path
        Directory to save the plot.
    bin_centers : ndarray, optional
        ADC bin centers for x-axis of panel 3.  If None, uses bin index.
    """
    out_dir = Path(out_dir)
    r = crossval_result

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # -- Panel 1: DNL factor scatter --
    ax = axes[0]
    if r.dnl_factors_A is not None and r.dnl_factors_B is not None:
        fa, fb = r.dnl_factors_A, r.dnl_factors_B
        if fa.shape == fb.shape:
            valid = (fa != 1.0) | (fb != 1.0)
            if np.any(valid):
                ax.scatter(fa[valid], fb[valid], s=4, alpha=0.4, c="steelblue",
                           edgecolors="none")
                lims = [
                    min(fa[valid].min(), fb[valid].min()) - 0.02,
                    max(fa[valid].max(), fb[valid].max()) + 0.02,
                ]
                ax.plot(lims, lims, "--", color="grey", linewidth=0.8, alpha=0.6)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
            corr_str = f"r = {r.dnl_correlation:.3f}" if not np.isnan(r.dnl_correlation) else "r = N/A"
            ax.set_title(f"DNL Factor Correlation ({corr_str})", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Shape mismatch", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title("DNL Factor Correlation", fontsize=9)
    else:
        ax.text(0.5, 0.5, "DNL factors unavailable", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title("DNL Factor Correlation", fontsize=9)
    ax.set_xlabel("DNL factors (half A)", fontsize=8)
    ax.set_ylabel("DNL factors (half B)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")

    # ── Panel 2: NLL bar chart ──────────────────────────────────
    ax = axes[1]
    labels = ["No DNL", "Self DNL", "Cross DNL"]
    vals_A = [r.nll_no_dnl_A, r.nll_self_dnl_A, r.nll_cross_dnl_A]
    vals_B = [r.nll_no_dnl_B, r.nll_self_dnl_B, r.nll_cross_dnl_B]

    # Normalise to no-DNL baseline for readability (show delta-NLL)
    def _delta(vals):
        base = vals[0]
        if base is None or np.isnan(base):
            return [0.0] * len(vals)
        return [float(v - base) if v is not None and not np.isnan(v) else 0.0
                for v in vals]

    delta_A = _delta(vals_A)
    delta_B = _delta(vals_B)

    x = np.arange(len(labels))
    w = 0.35
    bars_a = ax.bar(x - w / 2, delta_A, w, label="Half A", color="steelblue",
                    alpha=0.8)
    bars_b = ax.bar(x + w / 2, delta_B, w, label="Half B", color="coral",
                    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\Delta$NLL (vs no-DNL)", fontsize=8)
    ax.set_title("NLL Improvement from DNL", fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)

    # Annotate bars
    for bars in [bars_a, bars_b]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.0f}", ha="center",
                        va="bottom" if h > 0 else "top",
                        fontsize=6, color="black")

    # ── Panel 3: Verdict summary ────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    verdict_color = {
        "hardware_signal": "#2e7d32",
        "mixed": "#f57f17",
        "overfitting": "#c62828",
        "unknown": "#757575",
        "insufficient_data": "#757575",
    }
    vc = verdict_color.get(r.verdict, "#757575")

    lines = [
        f"Verdict: {r.verdict.replace('_', ' ').upper()}",
        "",
    ]
    lines.extend(r.verdict_reasons)
    lines.append("")
    if not np.isnan(r.self_improvement_A):
        lines.append(f"Self-DNL improvement:  A={r.self_improvement_A:.1f}  B={r.self_improvement_B:.1f}")
    if not np.isnan(r.cross_improvement_A):
        lines.append(f"Cross-DNL improvement: A={r.cross_improvement_A:.1f}  B={r.cross_improvement_B:.1f}")
    if not np.isnan(r.overfitting_indicator_A):
        lines.append(f"Overfitting indicator: A={r.overfitting_indicator_A:.1f}  B={r.overfitting_indicator_B:.1f}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=vc, alpha=0.15))
    ax.set_title("DNL Cross-Validation Summary", fontsize=10)

    fig.tight_layout()
    out_path = out_dir / "dnl_crossval.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved DNL cross-validation plot to %s", out_path)


# ═══════════════════════════════════════════════════════════════════════
# C1: Code-domain diagnostics
# ═══════════════════════════════════════════════════════════════════════

def compute_code_domain_diagnostics(
    fit_params: Mapping[str, object],
    min_counts: float = 5.0,
    dnl_period: int = 64,
) -> dict:
    """Compute DNL-specific residual diagnostics in raw ADC-code space.

    DNL is a code-locked hardware effect: it depends on the ADC code
    number, not on energy.  This function computes DNL-specific metrics
    that complement the energy-domain pull diagnostics:

    - Code-phase residual structure (bin_index mod *dnl_period*)
    - Spectral power at the expected DNL frequency
    - Correlation between pulls and DNL correction factors

    Generic autocorrelation and periodogram statistics are already
    covered by ``compute_pull_diagnostics``; this function focuses on
    what is unique to the ADC code domain.

    Parameters
    ----------
    fit_params : dict
        Must contain ``_plot_hist``, ``_plot_model_total``, and
        optionally ``_dnl`` with ``dnl_factors``.
    min_counts : float
        Minimum model prediction to include a bin.
    dnl_period : int
        Expected DNL periodicity from ADC subranging (default 64).

    Returns
    -------
    dict
        Scalar metrics and ``_arrays`` for plotting.
    """
    model = np.asarray(fit_params.get("_plot_model_total", []), dtype=float)
    hist = np.asarray(fit_params.get("_plot_hist", []), dtype=float)
    if model.size == 0 or hist.size == 0:
        return {}

    n_bins = model.size
    mask = model > min_counts
    n = int(mask.sum())
    if n < 16:
        return {}

    # Raw bin indices (proxy for ADC code)
    bin_idx = np.arange(n_bins)[mask]
    pulls = (hist[mask] - model[mask]) / np.sqrt(np.maximum(model[mask], 1.0))

    result: dict = {"n_bins": n}

    # ── Code-phase residual (mod dnl_period) ──────────────────────────
    # Mean pull per phase bin — reveals periodic hardware structure
    phases = bin_idx % dnl_period
    phase_mean = np.full(dnl_period, np.nan)
    phase_count = np.zeros(dnl_period, dtype=int)
    for ph in range(dnl_period):
        sel = phases == ph
        if sel.sum() >= 2:
            phase_mean[ph] = float(np.mean(pulls[sel]))
            phase_count[ph] = int(sel.sum())
    valid_phases = np.isfinite(phase_mean)
    if valid_phases.sum() > 0:
        result["code_phase_pull_rms"] = round(
            float(np.sqrt(np.nanmean(phase_mean[valid_phases] ** 2))), 4
        )
        result["code_phase_pull_max"] = round(
            float(np.nanmax(np.abs(phase_mean[valid_phases]))), 4
        )
        result["code_phase_n_valid"] = int(valid_phases.sum())

    # ── ACF and periodogram (kept for plotting) ───────────────────────
    pull_centered = pulls - np.mean(pulls)
    var = float(np.var(pulls))
    max_lag = min(50, n // 4)
    acf = np.zeros(max_lag)
    if var > 0:
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = float(
                    np.mean(pull_centered[:-lag] * pull_centered[lag:])
                ) / var

    N = len(pull_centered)
    fft_vals = sp_fft.rfft(pull_centered)
    power = np.abs(fft_vals) ** 2 / N
    freqs = sp_fft.rfftfreq(N)  # cycles per bin

    # ── DNL-specific: power at expected DNL frequency ─────────────────
    if len(power) > 1:
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]
        mean_p = float(np.mean(power_no_dc))
        if dnl_period > 0 and N > dnl_period and mean_p > 0:
            target_freq = 1.0 / dnl_period
            idx_target = int(np.argmin(np.abs(freqs_no_dc - target_freq)))
            lo_i = max(0, idx_target - 2)
            hi_i = min(len(power_no_dc), idx_target + 3)
            peak_power = float(np.max(power_no_dc[lo_i:hi_i]))
            result["code_fft_power_at_dnl_period"] = round(peak_power, 4)
            result["code_fft_dnl_period_ratio"] = round(
                peak_power / mean_p, 2
            )

    # ── DNL factor diagnostics (if available) ─────────────────────────
    dnl_meta = fit_params.get("_dnl", {})
    dnl_factors = dnl_meta.get("dnl_factors")
    if dnl_factors is not None:
        dnl_arr = np.asarray(dnl_factors, dtype=float)
        if dnl_arr.size == n_bins:
            dnl_masked = dnl_arr[mask]
            result["dnl_pull_correlation"] = round(
                float(np.corrcoef(pulls, dnl_masked - 1.0)[0, 1])
                if len(pulls) > 2 else 0.0,
                4,
            )

    # ── Stash arrays for plotting ─────────────────────────────────────
    result["_arrays"] = {
        "bin_idx": bin_idx,
        "pulls": pulls,
        "phases": phases,
        "phase_mean": phase_mean,
        "acf": acf,
        "power": power,
        "freqs": freqs,
    }

    return result


def plot_code_domain_diagnostics(
    diag_result: dict,
    out_dir: str | Path,
    dnl_period: int = 64,
) -> None:
    """Plot code-domain residual diagnostics (4 panels).

    Panel 1: Pull vs raw ADC bin index
    Panel 2: Mean pull vs code phase (mod *dnl_period*)
    Panel 3: Code-domain ACF
    Panel 4: Code-domain periodogram
    """
    out_dir = Path(out_dir)
    arrays = diag_result.get("_arrays")
    if arrays is None:
        return

    bin_idx = arrays["bin_idx"]
    pulls = arrays["pulls"]
    phase_mean = arrays["phase_mean"]
    acf = arrays["acf"]
    power = arrays["power"]
    freqs = arrays["freqs"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: Pull vs ADC bin index
    ax = axes[0, 0]
    ax.scatter(bin_idx, pulls, s=1.5, alpha=0.6, color="steelblue", rasterized=True)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(2, color="orange", lw=0.5, ls="--")
    ax.axhline(-2, color="orange", lw=0.5, ls="--")
    ax.set_xlabel("ADC bin index (code number)")
    ax.set_ylabel("Pull (data−model)/√model")
    ax.set_title("Residual vs Raw ADC Code")

    # Panel 2: Mean pull vs code phase
    ax = axes[0, 1]
    valid = np.isfinite(phase_mean)
    ph_idx = np.arange(len(phase_mean))
    colors = np.where(phase_mean[valid] > 0, "tomato", "steelblue")
    ax.bar(ph_idx[valid], phase_mean[valid], color=colors, width=0.8, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    rms = diag_result.get("code_phase_pull_rms", 0)
    ax.set_xlabel(f"Code phase (mod {dnl_period})")
    ax.set_ylabel("Mean pull")
    ax.set_title(f"Residual vs Code Phase (RMS={rms:.3f})")

    # Panel 3: Code-domain ACF
    ax = axes[1, 0]
    lags = np.arange(len(acf))
    ax.bar(lags, acf, color="steelblue", width=0.8, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    # 95% confidence band for white noise: ±1.96/√N
    n = len(pulls)
    ci = 1.96 / np.sqrt(n) if n > 0 else 0
    ax.axhline(ci, color="orange", lw=0.5, ls="--")
    ax.axhline(-ci, color="orange", lw=0.5, ls="--")
    ax.set_xlabel("Lag (bins)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Code-Domain ACF")
    ax.set_xlim(-0.5, min(30, len(acf)) - 0.5)

    # Panel 4: Code-domain periodogram (log-log)
    ax = axes[1, 1]
    if len(power) > 1:
        f_plot = freqs[1:]
        p_plot = power[1:]
        ax.loglog(f_plot, p_plot, color="steelblue", lw=0.5, alpha=0.6)
        mean_p = float(np.mean(p_plot))
        ax.axhline(mean_p, color="k", lw=0.5, ls="--", label=f"Mean={mean_p:.2f}")
        # Log-spaced smoothing
        if len(p_plot) > 20:
            n_sb = min(30, len(p_plot) // 3)
            log_edges = np.geomspace(f_plot[0], f_plot[-1], n_sb + 1)
            sf, sp = [], []
            for i in range(n_sb):
                m = (f_plot >= log_edges[i]) & (f_plot < log_edges[i + 1])
                if np.any(m):
                    sf.append(np.sqrt(log_edges[i] * log_edges[i + 1]))
                    sp.append(np.mean(p_plot[m]))
            if sf:
                ax.loglog(sf, sp, color="steelblue", lw=1.8, alpha=0.9)
        # Mark expected DNL period
        if dnl_period > 0:
            target_freq = 1.0 / dnl_period
            ax.axvline(
                target_freq, color="red", lw=0.8, ls="--",
                label=f"1/{dnl_period} cycles/bin",
            )
        ax.legend(fontsize=7)
        ax.set_xlim(f_plot[0] * 0.8, 0.5)
    ax.set_xlabel("Frequency (cycles/bin)")
    ax.set_ylabel("Power")
    ax.set_title("Code-Domain Periodogram")

    fig.suptitle("Code-Domain Residual Diagnostics (C1)", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fit_code_domain_diagnostics.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved code-domain diagnostics to %s", out_path)


# ═══════════════════════════════════════════════════════════════════════
# C3: Local peak-window signed-bias metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_signed_bias_metrics(
    fit_params: Mapping[str, object],
    min_counts: float = 5.0,
) -> dict:
    """Compute signed-bias, area-bias, run-length, and asymmetry metrics
    per isotope peak window.

    These aggregate statistics are valid under both DNL-corrected and
    uncorrected models because they summarise collective residual
    behaviour rather than individual bin pulls.

    Metrics per peak window
    -----------------------
    signed_bias : mean(data−model) / mean(model)
        Fractional systematic over/undershoot.
    area_bias_sigma : (Σdata − Σmodel) / √(Σmodel)
        Number of σ the total area is off.
    max_run_length : longest consecutive run of same-sign residuals
        Catches coherent structure within the peak.
    lo_negative_frac : fraction of low-energy-side bins with negative
        residuals (asymmetry misfit indicator).
    """
    model = np.asarray(fit_params.get("_plot_model_total", []), dtype=float)
    hist = np.asarray(fit_params.get("_plot_hist", []), dtype=float)
    centers = np.asarray(fit_params.get("_plot_centers", []), dtype=float)
    if model.size == 0 or hist.size == 0 or centers.size == 0:
        return {}

    result: dict = {}

    for iso, (lo, hi) in _PEAK_REGIONS.items():
        rmask = (centers >= lo) & (centers <= hi) & (model > min_counts)
        n_r = int(rmask.sum())
        if n_r < 5:
            continue

        d = hist[rmask]
        m = model[rmask]
        c = centers[rmask]
        resid = d - m

        # Signed bias (fractional)
        mean_m = float(np.mean(m))
        signed_bias = float(np.mean(resid)) / mean_m if mean_m > 0 else 0.0
        result[f"{iso}_signed_bias"] = round(signed_bias, 6)

        # Area bias in sigma
        sum_d = float(np.sum(d))
        sum_m = float(np.sum(m))
        area_bias = (sum_d - sum_m) / np.sqrt(max(sum_m, 1.0))
        result[f"{iso}_area_bias_sigma"] = round(float(area_bias), 3)

        # Longest same-sign run
        signs = np.sign(resid)
        run_lengths = []
        current_run = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i - 1] and signs[i] != 0:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        result[f"{iso}_max_run_length"] = int(max(run_lengths))
        result[f"{iso}_n_bins"] = n_r

        # Low-energy-side negative fraction (asymmetry indicator)
        # Split the window at the peak centroid (middle of window)
        mid_e = (lo + hi) / 2.0
        lo_side = (c < mid_e)
        if lo_side.sum() >= 2:
            lo_neg = float(np.mean(resid[lo_side] < 0))
            result[f"{iso}_lo_negative_frac"] = round(lo_neg, 3)

    return result


# ═══════════════════════════════════════════════════════════════════════
# C2: Covariance-aware (whitened) residuals for self-estimated DNL
# ═══════════════════════════════════════════════════════════════════════

def compute_whitened_residuals(
    fit_params: Mapping[str, object],
    min_counts: float = 5.0,
) -> dict:
    """Compute whitened residuals that account for smoothing covariance
    introduced by self-estimated band-pass DNL correction.

    The band-pass filter (``uniform_filter1d`` with window *W*) creates
    inter-bin covariance because neighbouring bins share smoothed
    information.  The approximate covariance structure of the corrected
    counts is:

        Cov(n_i, n_j) ≈ δ_{ij} · n_i  +  n_i · n_j · K(|i-j|)

    where K is the triangular autocorrelation of the uniform kernel:

        K(lag) = max(0, (W - |lag|) / W²)

    Whitening via Cholesky decomposition produces residuals that are
    approximately N(0, 1) under the correct model even when naive pulls
    are correlated.

    Only computed when ``operator_class == "self_estimated_bandpass"``.
    """
    dnl_meta = fit_params.get("_dnl", {})
    if dnl_meta.get("operator_class") != "self_estimated_bandpass":
        return {}

    model = np.asarray(fit_params.get("_plot_model_total", []), dtype=float)
    hist = np.asarray(fit_params.get("_plot_hist", []), dtype=float)
    if model.size == 0 or hist.size == 0:
        return {}

    W = int(dnl_meta.get("dnl_smooth_window", 61))
    mask = model > min_counts
    n = int(mask.sum())
    if n < 20:
        return {}

    d = hist[mask]
    m = model[mask]

    # ── Build approximate covariance matrix ───────────────────────────
    # Poisson diagonal + smoothing-induced off-diagonal
    C = np.diag(m.copy())  # Poisson variance on diagonal
    for i in range(n):
        for j in range(i + 1, min(i + W, n)):
            lag = j - i
            k_val = max(0.0, (W - lag) / (W * W))
            off_diag = m[i] * m[j] * k_val
            C[i, j] += off_diag
            C[j, i] += off_diag

    # ── Cholesky whitening ────────────────────────────────────────────
    try:
        L = np.linalg.cholesky(C)
        raw_resid = d - m
        whitened = np.linalg.solve(L, raw_resid)
    except np.linalg.LinAlgError:
        # Covariance not positive definite - add jitter
        jitter = 1e-6 * np.mean(np.diag(C))
        C += jitter * np.eye(n)
        try:
            L = np.linalg.cholesky(C)
            raw_resid = d - m
            whitened = np.linalg.solve(L, raw_resid)
        except np.linalg.LinAlgError:
            return {"error": "Cholesky decomposition failed"}

    result: dict = {"n_bins": n, "dnl_smooth_window": W}

    # ── Whitened statistics ───────────────────────────────────────────
    result["whitened_pull_mean"] = round(float(np.mean(whitened)), 4)
    result["whitened_pull_sigma"] = round(float(np.std(whitened, ddof=1)), 4)

    # Whitened Durbin-Watson
    ss = float(np.sum(whitened ** 2))
    if ss > 0:
        result["whitened_dw"] = round(
            float(np.sum(np.diff(whitened) ** 2) / ss), 4
        )

    # Whitened lag-1 ACF
    wc = whitened - np.mean(whitened)
    wvar = float(np.var(whitened))
    if wvar > 0 and n > 1:
        result["whitened_acf_lag1"] = round(
            float(np.mean(wc[:-1] * wc[1:])) / wvar, 4
        )

    # ── Naive (unwhitened) for comparison ─────────────────────────────
    naive_pulls = (d - m) / np.sqrt(np.maximum(m, 1.0))
    result["naive_pull_sigma"] = round(float(np.std(naive_pulls, ddof=1)), 4)
    nss = float(np.sum(naive_pulls ** 2))
    if nss > 0:
        result["naive_dw"] = round(
            float(np.sum(np.diff(naive_pulls) ** 2) / nss), 4
        )

    # ── Flag if they disagree ─────────────────────────────────────────
    w_sig = result.get("whitened_pull_sigma", 1.0)
    n_sig = result.get("naive_pull_sigma", 1.0)
    if abs(w_sig - n_sig) > 0.15:
        result["interpretation_note"] = (
            f"Whitened (σ={w_sig:.3f}) and naive (σ={n_sig:.3f}) pull "
            f"sigmas disagree by {abs(w_sig - n_sig):.3f}. The "
            f"whitened value is the correct one under self-estimated "
            f"band-pass DNL; the naive value is inflated by smoothing "
            f"covariance."
        )

    return result
