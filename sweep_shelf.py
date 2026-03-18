"""
Sweep f_shelf and sigma_shelf for Po218 and Po212 to find optimal values.

Phase 1: Sweep Po218 (keep Po212 at current)
Phase 2: Use best Po218, sweep Po212

Collects: chi2/ndf, pull_sigma, rebin10_pull_sigma, split-half max|z|, split-half result
"""
import yaml
import subprocess
import json
import os
import copy
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# Read base config
with open(CONFIG_PATH) as f:
    BASE_CONFIG = yaml.safe_load(f)


def set_shelf_params(cfg, isotope, f_shelf, sigma_shelf):
    """Set shelf prior means for a given isotope."""
    sf = cfg["spectral_fit"]
    sf[f"f_shelf_{isotope}_prior"][0] = f_shelf
    sf[f"sigma_shelf_{isotope}_prior"][0] = sigma_shelf


def run_pipeline(cfg, label):
    """Write config, run pipeline, return metrics dict."""
    tmp_config = os.path.join(BASE_DIR, "_sweep_config.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "analyze.py", "--config", tmp_config, "--output-dir", "results"],
        capture_output=True, text=True, cwd=BASE_DIR, timeout=600,
    )
    elapsed = time.time() - t0

    # Find the output directory from the last line
    out_dir = None
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if "Results written to" in line:
            out_dir = line.split("->")[-1].strip()
            break

    if not out_dir:
        print(f"  [{label}] FAILED after {elapsed:.0f}s — no output dir found")
        return None

    out_dir = os.path.join(BASE_DIR, out_dir)
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(f"  [{label}] FAILED — no summary.json")
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    sf = summary.get("spectral_fit", {})
    pd = sf.get("pull_diagnostics", {})
    sh = sf.get("split_half", {})

    metrics = {
        "label": label,
        "out_dir": os.path.basename(out_dir),
        "chi2_ndf": sf.get("chi2_ndf"),
        "pull_sigma": pd.get("pull_sigma"),
        "rebin10_pull_sigma": pd.get("rebin10_pull_sigma"),
        "rebin10_dw": pd.get("rebin10_durbin_watson"),
        "rebin10_acf": pd.get("rebin10_autocorr_lag1"),
        "rebin10_ks": pd.get("rebin10_ks_pvalue"),
        "split_half_max_z": sh.get("max_abs_z"),
        "split_half_result": sh.get("result"),
        "elapsed_s": elapsed,
    }
    print(f"  [{label}] chi2={metrics['chi2_ndf']:.4f}  pull_s={metrics['pull_sigma']:.3f}  "
          f"rb10={metrics['rebin10_pull_sigma']:.3f}  sh_z={metrics.get('split_half_max_z', '?')}  "
          f"({elapsed:.0f}s)")
    return metrics


def print_table(results, title):
    """Print a formatted results table."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    header = f"{'Label':<30} {'chi2/ndf':>8} {'pull_σ':>8} {'rb10_σ':>8} {'rb10_DW':>8} {'rb10_KS':>8} {'SH_z':>6} {'SH':>5}"
    print(header)
    print("-" * 90)
    for r in results:
        if r is None:
            continue
        sh_z = f"{r['split_half_max_z']:.2f}" if r.get('split_half_max_z') else "?"
        sh_r = r.get('split_half_result', '?') or '?'
        print(f"{r['label']:<30} {r['chi2_ndf']:>8.4f} {r['pull_sigma']:>8.3f} "
              f"{r['rebin10_pull_sigma']:>8.3f} {r['rebin10_dw']:>8.3f} "
              f"{r['rebin10_ks']:>8.4f} {sh_z:>6} {sh_r:>5}")
    print("=" * 90)


def main():
    # Current values (baseline)
    current_po218_f = 0.12
    current_po218_s = 0.35
    current_po212_f = 0.12
    current_po212_s = 0.35

    # Sweep grid
    f_shelf_values = [0.15, 0.20, 0.25, 0.30]
    sigma_shelf_fixed = 0.40  # slightly wider than current 0.35

    # =========================================================
    # Phase 1: Sweep Po218 f_shelf (keep Po212 at current)
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Sweeping Po218 f_shelf")
    print(f"  Po212 fixed at f={current_po212_f}, sigma={current_po212_s}")
    print(f"  Po218 sigma_shelf fixed at {sigma_shelf_fixed}")
    print("=" * 60)

    po218_results = []
    for f_shelf in f_shelf_values:
        cfg = copy.deepcopy(BASE_CONFIG)
        set_shelf_params(cfg, "Po218", f_shelf, sigma_shelf_fixed)
        set_shelf_params(cfg, "Po212", current_po212_f, current_po212_s)
        label = f"Po218 f={f_shelf:.2f} s={sigma_shelf_fixed:.2f}"
        metrics = run_pipeline(cfg, label)
        if metrics:
            metrics["f_shelf"] = f_shelf
            metrics["sigma_shelf"] = sigma_shelf_fixed
        po218_results.append(metrics)

    print_table(po218_results, "Po218 f_shelf Sweep (sigma_shelf=0.40)")

    # Find best Po218 by minimum rebin10_pull_sigma that still has split-half PASS
    valid_po218 = [r for r in po218_results if r and r.get("split_half_result") == "PASS"]
    if not valid_po218:
        # Fall back to all results
        valid_po218 = [r for r in po218_results if r]

    if valid_po218:
        # Pick the one with rebin10_pull_sigma closest to 1.0 (but prefer < 2.0)
        best_po218 = min(valid_po218, key=lambda r: abs(r["rebin10_pull_sigma"] - 1.0))
        best_po218_f = best_po218["f_shelf"]
        best_po218_s = best_po218["sigma_shelf"]
        print(f"\n  Best Po218: f_shelf={best_po218_f}, sigma_shelf={best_po218_s} "
              f"(rb10={best_po218['rebin10_pull_sigma']:.3f})")
    else:
        best_po218_f = 0.20
        best_po218_s = sigma_shelf_fixed
        print(f"\n  No valid results, using default Po218: f_shelf={best_po218_f}")

    # =========================================================
    # Phase 2: Sweep Po212 f_shelf (with best Po218)
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Sweeping Po212 f_shelf")
    print(f"  Po218 fixed at f={best_po218_f}, sigma={best_po218_s}")
    print(f"  Po212 sigma_shelf fixed at {sigma_shelf_fixed}")
    print("=" * 60)

    po212_results = []
    for f_shelf in f_shelf_values:
        cfg = copy.deepcopy(BASE_CONFIG)
        set_shelf_params(cfg, "Po218", best_po218_f, best_po218_s)
        set_shelf_params(cfg, "Po212", f_shelf, sigma_shelf_fixed)
        label = f"Po212 f={f_shelf:.2f} s={sigma_shelf_fixed:.2f}"
        metrics = run_pipeline(cfg, label)
        if metrics:
            metrics["f_shelf"] = f_shelf
            metrics["sigma_shelf"] = sigma_shelf_fixed
        po212_results.append(metrics)

    print_table(po212_results, f"Po212 f_shelf Sweep (sigma_shelf=0.40, Po218 f={best_po218_f})")

    # Find best Po212
    valid_po212 = [r for r in po212_results if r and r.get("split_half_result") == "PASS"]
    if not valid_po212:
        valid_po212 = [r for r in po212_results if r]

    if valid_po212:
        best_po212 = min(valid_po212, key=lambda r: abs(r["rebin10_pull_sigma"] - 1.0))
        best_po212_f = best_po212["f_shelf"]
        best_po212_s = best_po212["sigma_shelf"]
        print(f"\n  Best Po212: f_shelf={best_po212_f}, sigma_shelf={best_po212_s} "
              f"(rb10={best_po212['rebin10_pull_sigma']:.3f})")

    # =========================================================
    # Final summary
    # =========================================================
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    if valid_po218:
        print(f"  Po218: f_shelf={best_po218_f:.2f}, sigma_shelf={best_po218_s:.2f}")
    if valid_po212:
        print(f"  Po212: f_shelf={best_po212_f:.2f}, sigma_shelf={best_po212_s:.2f}")
    print("=" * 60)

    # Clean up temp config
    tmp_config = os.path.join(BASE_DIR, "_sweep_config.yaml")
    if os.path.exists(tmp_config):
        os.remove(tmp_config)


if __name__ == "__main__":
    main()
