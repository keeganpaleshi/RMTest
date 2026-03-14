"""Spectral fit diagnostic plots: correlation matrix, pull histogram, parameter table."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


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
    ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)

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
            "ndf_effective", "chi2_ndf_effective"}
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
    bars = ax.barh(range(len(rows)), vals_plot, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names_plot, fontsize=7)
    ax.set_xlabel("Relative uncertainty (%)")
    ax.set_title("Parameter Relative Uncertainties", fontsize=11)
    ax.axvline(10, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add value labels
    for bar, v in zip(bars, vals_plot):
        x_pos = bar.get_width() + 0.5
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
    ax.set_xlabel("z-score (half A \u2212 half B)")

    # Reference lines
    for z_ref in [-2, -1, 1, 2]:
        ax.axvline(z_ref, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    # Shade bands
    ax.axvspan(-1, 1, alpha=0.06, color="green")
    ax.axvspan(-2, 2, alpha=0.03, color="orange")

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

    # Anderson-Darling (tests normality — more sensitive in tails than KS)
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
    for rebin_factor in [10, 30]:
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
      (a) Raw pull autocorrelation (bin-level — dominated by ADC DNL)
      (b) Rebinned pull autocorrelation (×10 — model-level structure)
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

    # Compute rebinned pulls for panel (b)
    rebin = 10
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
    ax.set_title("(a) Raw ACF (bin-level — DNL dominated)", fontsize=9)
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
        ["DW (bin)", _fv("durbin_watson"), "2.00", "1.5–2.5"],
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
    r10_dw = d.get("rebin10_durbin_watson", "?")
    r10_ac = d.get("rebin10_autocorr_lag1", "?")
    r10_rz = d.get("rebin10_runs_test_z", "?")
    r10_ks = d.get("rebin10_ks_pvalue", "?")
    r10_sig = d.get("rebin10_pull_sigma", "?")
    r30_dw = d.get("rebin30_durbin_watson", "?")
    r30_ac = d.get("rebin30_autocorr_lag1", "?")
    r30_rz = d.get("rebin30_runs_test_z", "?")

    def _safe_fmt(v, fmt=".3f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)

    rows_f = [
        ["σ (×10 rebin)", _safe_fmt(r10_sig), "1.00", "< 1.2"],
        ["DW (×10 rebin)", _safe_fmt(r10_dw), "2.00", "1.5–2.5"],
        ["ACF lag-1 (×10)", _safe_fmt(r10_ac), "0.00", "|r| < 0.15"],
        ["Runs z (×10)", _safe_fmt(r10_rz, ".1f"), "0", "|z| < 3"],
        ["KS p (×10)", _safe_fmt(r10_ks, ".4f"), "> 0.05", "> 0.01"],
        ["DW (×30 rebin)", _safe_fmt(r30_dw), "2.00", "1.5–2.5"],
        ["ACF lag-1 (×30)", _safe_fmt(r30_ac), "0.00", "|r| < 0.15"],
        ["Runs z (×30)", _safe_fmt(r30_rz, ".1f"), "0", "|z| < 3"],
    ]
    checks_f = [
        (r10_sig < 1.2) if isinstance(r10_sig, (int, float)) else False,
        (1.5 < r10_dw < 2.5) if isinstance(r10_dw, (int, float)) else False,
        (abs(r10_ac) < 0.15) if isinstance(r10_ac, (int, float)) else False,
        (abs(r10_rz) < 3) if isinstance(r10_rz, (int, float)) else False,
        (r10_ks > 0.01) if isinstance(r10_ks, (int, float)) else False,
        (1.5 < r30_dw < 2.5) if isinstance(r30_dw, (int, float)) else False,
        (abs(r30_ac) < 0.15) if isinstance(r30_ac, (int, float)) else False,
        (abs(r30_rz) < 3) if isinstance(r30_rz, (int, float)) else False,
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
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 1.0 + 0.6 * n)))

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
        "—",
        "—",
        "—",
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
        f"Model Complexity Comparison — {conclusion.replace('_', ' ').title()}",
        fontsize=12, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "fit_model_comparison.png", dpi=200)
    plt.close(fig)
    logger.info(
        "Saved model comparison to %s",
        out_dir / "fit_model_comparison.png",
    )
