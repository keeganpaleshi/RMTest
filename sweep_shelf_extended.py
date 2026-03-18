"""
Extended sweep: push f_shelf much further for Po218 and Po212.
Also varies sigma_shelf to find optimal transition width.

Focus: LOW-ENERGY side quality only.
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

with open(CONFIG_PATH) as f:
    BASE_CONFIG = yaml.safe_load(f)


def set_shelf_params(cfg, isotope, f_shelf, sigma_shelf):
    sf = cfg["spectral_fit"]
    sf[f"f_shelf_{isotope}_prior"][0] = f_shelf
    sf[f"sigma_shelf_{isotope}_prior"][0] = sigma_shelf


def run_pipeline(cfg, label):
    tmp_config = os.path.join(BASE_DIR, "_sweep_config.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "analyze.py", "--config", tmp_config, "--output-dir", "results"],
        capture_output=True, text=True, cwd=BASE_DIR, timeout=600,
    )
    elapsed = time.time() - t0

    out_dir = None
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if "Results written to" in line:
            out_dir = line.split("->")[-1].strip()
            break

    if not out_dir:
        print(f"  [{label}] FAILED after {elapsed:.0f}s", flush=True)
        return None

    out_dir = os.path.join(BASE_DIR, out_dir)
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(f"  [{label}] FAILED — no summary.json", flush=True)
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    sf = summary.get("spectral_fit", {})
    pd = sf.get("pull_diagnostics", {})

    # Try to get split-half info from the log output
    sh_max_z = None
    sh_result = None
    for line in (result.stdout + result.stderr).splitlines():
        if "Split-half validation" in line and "shape params" in line:
            # Parse: "Split-half validation (shape params): max|z|=0.57, ... -> PASS"
            try:
                parts = line.split("max|z|=")[1]
                sh_max_z = float(parts.split(",")[0])
                sh_result = line.strip().split("-> ")[-1]
            except:
                pass

    metrics = {
        "label": label,
        "out_dir": os.path.basename(out_dir),
        "chi2_ndf": sf.get("chi2_ndf"),
        "pull_sigma": pd.get("pull_sigma"),
        "rebin10_pull_sigma": pd.get("rebin10_pull_sigma"),
        "rebin10_dw": pd.get("rebin10_durbin_watson"),
        "rebin10_acf": pd.get("rebin10_autocorr_lag1"),
        "rebin10_ks": pd.get("rebin10_ks_pvalue"),
        "sh_max_z": sh_max_z,
        "sh_result": sh_result,
        "elapsed_s": elapsed,
    }
    sh_str = f"{sh_max_z:.2f} {sh_result}" if sh_max_z is not None else "?"
    print(f"  [{label}] chi2={metrics['chi2_ndf']:.4f}  rb10={metrics['rebin10_pull_sigma']:.3f}  "
          f"DW={metrics['rebin10_dw']:.3f}  KS={metrics['rebin10_ks']:.3f}  SH={sh_str}  "
          f"({elapsed:.0f}s)", flush=True)
    return metrics


def print_table(results, title):
    print(f"\n{'='*100}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*100}", flush=True)
    header = (f"{'Label':<35} {'chi2':>7} {'pull_σ':>7} {'rb10_σ':>7} "
              f"{'rb10_DW':>7} {'rb10_KS':>7} {'SH_z':>6} {'SH':>5}")
    print(header, flush=True)
    print("-" * 100, flush=True)
    for r in results:
        if r is None:
            continue
        sh_z = f"{r['sh_max_z']:.2f}" if r.get('sh_max_z') is not None else "?"
        sh_r = r.get('sh_result', '?') or '?'
        print(f"{r['label']:<35} {r['chi2_ndf']:>7.4f} {r['pull_sigma']:>7.3f} "
              f"{r['rebin10_pull_sigma']:>7.3f} {r['rebin10_dw']:>7.3f} "
              f"{r['rebin10_ks']:>7.4f} {sh_z:>6} {sh_r:>5}", flush=True)
    print("=" * 100, flush=True)


def main():
    # =========================================================
    # Phase 1: Extended Po218 sweep (Po212 at f=0.25, sigma=0.40)
    # =========================================================
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1: Extended Po218 f_shelf sweep", flush=True)
    print("  Po212 fixed at f=0.25, sigma=0.40", flush=True)
    print("=" * 60, flush=True)

    po218_configs = [
        # (f_shelf, sigma_shelf)
        (0.30, 0.40),
        (0.35, 0.40),
        (0.40, 0.40),
        (0.50, 0.40),
        (0.35, 0.50),
        (0.40, 0.50),
        (0.50, 0.50),
    ]

    po218_results = []
    for f_shelf, sigma_shelf in po218_configs:
        cfg = copy.deepcopy(BASE_CONFIG)
        set_shelf_params(cfg, "Po218", f_shelf, sigma_shelf)
        set_shelf_params(cfg, "Po212", 0.25, 0.40)
        label = f"Po218 f={f_shelf:.2f} s={sigma_shelf:.2f}"
        metrics = run_pipeline(cfg, label)
        if metrics:
            metrics["f_shelf"] = f_shelf
            metrics["sigma_shelf"] = sigma_shelf
        po218_results.append(metrics)

    print_table(po218_results, "Po218 Extended Sweep")

    # =========================================================
    # Phase 2: Extended Po212 sweep (Po218 at f=0.25, sigma=0.40)
    # Note: using Po218 at 0.25 as baseline since we don't know
    # optimal Po218 yet — user can combine later
    # =========================================================
    print("\n" + "=" * 60, flush=True)
    print("PHASE 2: Extended Po212 f_shelf sweep", flush=True)
    print("  Po218 fixed at f=0.25, sigma=0.40", flush=True)
    print("=" * 60, flush=True)

    po212_configs = [
        (0.30, 0.40),
        (0.35, 0.40),
        (0.40, 0.40),
        (0.50, 0.40),
        (0.35, 0.50),
        (0.40, 0.50),
        (0.50, 0.50),
        (0.50, 0.60),
    ]

    po212_results = []
    for f_shelf, sigma_shelf in po212_configs:
        cfg = copy.deepcopy(BASE_CONFIG)
        set_shelf_params(cfg, "Po218", 0.25, 0.40)
        set_shelf_params(cfg, "Po212", f_shelf, sigma_shelf)
        label = f"Po212 f={f_shelf:.2f} s={sigma_shelf:.2f}"
        metrics = run_pipeline(cfg, label)
        if metrics:
            metrics["f_shelf"] = f_shelf
            metrics["sigma_shelf"] = sigma_shelf
        po212_results.append(metrics)

    print_table(po212_results, "Po212 Extended Sweep")

    # Clean up
    tmp_config = os.path.join(BASE_DIR, "_sweep_config.yaml")
    if os.path.exists(tmp_config):
        os.remove(tmp_config)

    print("\nDone! Review the tables above to pick optimal values.", flush=True)


if __name__ == "__main__":
    main()
