"""
Overnight halo parameter sweep for Po218, Po214, Po210 high-energy sides.

Phases 1-6: Sequential grid sweeps for each isotope
Phase 7: Combined verification
Phases 8-10: Fine-grained sweeps near optima

All results saved to sweep_halo_results.csv after every run.
"""
import yaml
import subprocess
import json
import os
import copy
import sys
import time
import csv
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
CSV_PATH = os.path.join(BASE_DIR, "sweep_halo_results.csv")

with open(CONFIG_PATH) as f:
    BASE_CONFIG = yaml.safe_load(f)

CSV_FIELDS = [
    "phase", "label", "isotope",
    "f_halo", "sigma_halo", "tau_halo",
    "chi2_ndf", "pull_sigma",
    "rebin10_pull_sigma", "rebin10_dw", "rebin10_acf", "rebin10_ks", "rebin10_runs_z",
    "sh_max_z", "sh_result",
    "S_Po210", "S_Po218", "S_Po214", "S_Po212",
    "mu_Po210", "mu_Po218", "mu_Po214", "mu_Po212",
    "sigma_Po210", "sigma_Po218", "sigma_Po214",
    "dS_Po210", "dS_Po218", "dS_Po214", "dS_Po212",
    "out_dir", "elapsed_s",
]


def set_halo_params(cfg, isotope, f_halo=None, sigma_halo=None, tau_halo=None):
    sf = cfg["spectral_fit"]
    if f_halo is not None:
        sf[f"f_halo_{isotope}_prior"][0] = f_halo
    if sigma_halo is not None:
        sf[f"sigma_halo_{isotope}_prior"][0] = sigma_halo
    if tau_halo is not None:
        sf[f"tau_halo_{isotope}_prior"][0] = tau_halo


def run_pipeline(cfg, label):
    tmp_config = os.path.join(BASE_DIR, "_sweep_halo_config.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "analyze.py", "--config", tmp_config, "--output-dir", "results"],
            capture_output=True, text=True, cwd=BASE_DIR, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMEOUT after 600s", flush=True)
        return None
    except Exception as e:
        print(f"  [{label}] ERROR: {e}", flush=True)
        return None
    elapsed = time.time() - t0

    out_dir = None
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if "Results written to" in line:
            out_dir = line.split("->")[-1].strip()
            break

    if not out_dir:
        print(f"  [{label}] FAILED after {elapsed:.0f}s — no output dir", flush=True)
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

    # Parse split-half from log
    sh_max_z = None
    sh_result = None
    for line in (result.stdout + result.stderr).splitlines():
        if "Split-half validation" in line and "shape params" in line:
            try:
                parts = line.split("max|z|=")[1]
                sh_max_z = float(parts.split(",")[0])
                sh_result = line.strip().split("-> ")[-1]
            except:
                pass

    # Extract fitted parameter values
    params = {}
    for iso in ["Po210", "Po218", "Po214", "Po212"]:
        for pname in ["S", "mu", "sigma", "dS"]:
            key = f"{pname}_{iso}"
            dkey = f"d{pname}_{iso}" if pname == "S" else None
            if pname == "dS":
                val = sf.get(f"dS_{iso}")
            else:
                val = sf.get(f"{pname}_{iso}")
            params[key] = val

    metrics = {
        "label": label,
        "out_dir": os.path.basename(out_dir),
        "chi2_ndf": sf.get("chi2_ndf"),
        "pull_sigma": pd.get("pull_sigma"),
        "rebin10_pull_sigma": pd.get("rebin10_pull_sigma"),
        "rebin10_dw": pd.get("rebin10_durbin_watson"),
        "rebin10_acf": pd.get("rebin10_autocorr_lag1"),
        "rebin10_ks": pd.get("rebin10_ks_pvalue"),
        "rebin10_runs_z": pd.get("rebin10_runs_test_z"),
        "sh_max_z": sh_max_z,
        "sh_result": sh_result,
        "elapsed_s": elapsed,
    }
    metrics.update(params)

    sh_str = f"{sh_max_z:.2f} {sh_result}" if sh_max_z is not None else "?"
    rb10 = metrics.get("rebin10_pull_sigma", 0) or 0
    chi2 = metrics.get("chi2_ndf", 0) or 0
    print(f"  [{label}] chi2={chi2:.4f} rb10={rb10:.3f} SH={sh_str} ({elapsed:.0f}s)", flush=True)
    return metrics


def append_csv(row):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def score(m):
    """Score a result. Lower is better. Returns (is_valid, score)."""
    if m is None:
        return (False, 999)
    if m.get("sh_result") != "PASS":
        return (False, 999)
    rb10 = m.get("rebin10_pull_sigma", 999) or 999
    acf = abs(m.get("rebin10_acf", 999) or 999)
    chi2 = m.get("chi2_ndf", 999) or 999
    ks = m.get("rebin10_ks", 0) or 0
    # Primary: rb10 close to 1.0
    s = abs(rb10 - 1.0)
    # Penalty for ACF out of range
    if acf > 0.15:
        s += (acf - 0.15) * 5
    # Penalty for chi2 far from 1.0
    s += abs(chi2 - 1.0) * 0.5
    # Bonus for good KS
    if ks > 0.05:
        s -= 0.1
    return (True, s)


def pick_best(results):
    valid = [(m, score(m)) for m in results if m is not None]
    valid.sort(key=lambda x: (not x[1][0], x[1][1]))
    if valid:
        return valid[0][0]
    return None


def main():
    total_runs = 0
    t_start = time.time()

    # Remove old CSV
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    # =========================================================
    # PHASE 1: Po218 sigma_halo x tau_halo grid
    # =========================================================
    print("\n" + "=" * 70, flush=True)
    print("PHASE 1: Po218 sigma_halo x tau_halo grid (f_halo=0.08)", flush=True)
    print("=" * 70, flush=True)

    sigma_vals = [0.18, 0.20, 0.22, 0.25]
    tau_vals = [0.02, 0.04, 0.06, 0.08]
    p1_results = []

    for sh in sigma_vals:
        for th in tau_vals:
            cfg = copy.deepcopy(BASE_CONFIG)
            set_halo_params(cfg, "Po218", sigma_halo=sh, tau_halo=th)
            label = f"Po218 sh={sh:.2f} th={th:.2f}"
            m = run_pipeline(cfg, label)
            if m:
                m["phase"] = 1
                m["isotope"] = "Po218"
                m["f_halo"] = 0.08
                m["sigma_halo"] = sh
                m["tau_halo"] = th
                append_csv(m)
            p1_results.append(m)
            total_runs += 1

    best_p1 = pick_best(p1_results)
    if best_p1:
        best_po218_sh = best_p1["sigma_halo"]
        best_po218_th = best_p1["tau_halo"]
        print(f"\n  PHASE 1 BEST: sigma_halo={best_po218_sh}, tau_halo={best_po218_th} "
              f"(rb10={best_p1['rebin10_pull_sigma']:.3f})", flush=True)
    else:
        best_po218_sh = 0.20
        best_po218_th = 0.04
        print(f"\n  PHASE 1: No PASS results, using defaults", flush=True)

    # =========================================================
    # PHASE 2: Po218 f_halo sweep
    # =========================================================
    print("\n" + "=" * 70, flush=True)
    print(f"PHASE 2: Po218 f_halo sweep (sh={best_po218_sh}, th={best_po218_th})", flush=True)
    print("=" * 70, flush=True)

    fh_vals = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    p2_results = []

    for fh in fh_vals:
        cfg = copy.deepcopy(BASE_CONFIG)
        set_halo_params(cfg, "Po218", f_halo=fh, sigma_halo=best_po218_sh, tau_halo=best_po218_th)
        label = f"Po218 fh={fh:.2f} sh={best_po218_sh:.2f} th={best_po218_th:.2f}"
        m = run_pipeline(cfg, label)
        if m:
            m["phase"] = 2
            m["isotope"] = "Po218"
            m["f_halo"] = fh
            m["sigma_halo"] = best_po218_sh
            m["tau_halo"] = best_po218_th
            append_csv(m)
        p2_results.append(m)
        total_runs += 1

    best_p2 = pick_best(p2_results)
    if best_p2:
        best_po218_fh = best_p2["f_halo"]
        print(f"\n  PHASE 2 BEST: f_halo={best_po218_fh} "
              f"(rb10={best_p2['rebin10_pull_sigma']:.3f})", flush=True)
    else:
        best_po218_fh = 0.08
        print(f"\n  PHASE 2: No PASS results, using default f_halo=0.08", flush=True)

    # Apply best Po218 to base for subsequent phases
    best_po218_cfg = copy.deepcopy(BASE_CONFIG)
    set_halo_params(best_po218_cfg, "Po218",
                    f_halo=best_po218_fh, sigma_halo=best_po218_sh, tau_halo=best_po218_th)

    # =========================================================
    # PHASE 3: Po214 sigma_halo x tau_halo grid
    # =========================================================
    print("\n" + "=" * 70, flush=True)
    print("PHASE 3: Po214 sigma_halo x tau_halo grid (f_halo=0.10)", flush=True)
    print(f"  Po218 fixed at fh={best_po218_fh}, sh={best_po218_sh}, th={best_po218_th}", flush=True)
    print("=" * 70, flush=True)

    sigma_vals_214 = [0.20, 0.22, 0.25]
    tau_vals_214 = [0.02, 0.04, 0.06, 0.08]
    p3_results = []

    for sh in sigma_vals_214:
        for th in tau_vals_214:
            cfg = copy.deepcopy(best_po218_cfg)
            set_halo_params(cfg, "Po214", sigma_halo=sh, tau_halo=th)
            label = f"Po214 sh={sh:.2f} th={th:.2f}"
            m = run_pipeline(cfg, label)
            if m:
                m["phase"] = 3
                m["isotope"] = "Po214"
                m["f_halo"] = 0.10
                m["sigma_halo"] = sh
                m["tau_halo"] = th
                append_csv(m)
            p3_results.append(m)
            total_runs += 1

    best_p3 = pick_best(p3_results)
    if best_p3:
        best_po214_sh = best_p3["sigma_halo"]
        best_po214_th = best_p3["tau_halo"]
        print(f"\n  PHASE 3 BEST: sigma_halo={best_po214_sh}, tau_halo={best_po214_th} "
              f"(rb10={best_p3['rebin10_pull_sigma']:.3f})", flush=True)
    else:
        best_po214_sh = 0.22
        best_po214_th = 0.06
        print(f"\n  PHASE 3: No PASS results, using defaults", flush=True)

    # =========================================================
    # PHASE 4: Po214 f_halo sweep
    # =========================================================
    print("\n" + "=" * 70, flush=True)
    print(f"PHASE 4: Po214 f_halo sweep (sh={best_po214_sh}, th={best_po214_th})", flush=True)
    print("=" * 70, flush=True)

    fh_vals_214 = [0.04, 0.06, 0.08, 0.10, 0.12]
    p4_results = []

    for fh in fh_vals_214:
        cfg = copy.deepcopy(best_po218_cfg)
        set_halo_params(cfg, "Po214", f_halo=fh, sigma_halo=best_po214_sh, tau_halo=best_po214_th)
        label = f"Po214 fh={fh:.2f} sh={best_po214_sh:.2f} th={best_po214_th:.2f}"
        m = run_pipeline(cfg, label)
        if m:
            m["phase"] = 4
            m["isotope"] = "Po214"
            m["f_halo"] = fh
            m["sigma_halo"] = best_po214_sh
            m["tau_halo"] = best_po214_th
            append_csv(m)
        p4_results.append(m)
        total_runs += 1

    best_p4 = pick_best(p4_results)
    if best_p4:
        best_po214_fh = best_p4["f_halo"]
        print(f"\n  PHASE 4 BEST: f_halo={best_po214_fh} "
              f"(rb10={best_p4['rebin10_pull_sigma']:.3f})", flush=True)
    else:
        best_po214_fh = 0.10
        print(f"\n  PHASE 4: No PASS results, using default f_halo=0.10", flush=True)

    # Apply best Po218+Po214 to base for subsequent phases
    best_218_214_cfg = copy.deepcopy(best_po218_cfg)
    set_halo_params(best_218_214_cfg, "Po214",
                    f_halo=best_po214_fh, sigma_halo=best_po214_sh, tau_halo=best_po214_th)

    # =========================================================
    # PHASE 5: Po210 sigma_halo x tau_halo grid
    # =========================================================
    print("\n" + "=" * 70, flush=True)
    print("PHASE 5: Po210 sigma_halo x tau_halo grid (f_halo=0.08)", flush=True)
    print(f"  Po218: fh={best_po218_fh}, sh={best_po218_sh}, th={best_po218_th}", flush=True)
    print(f"  Po214: fh={best_po214_fh}, sh={best_po214_sh}, th={best_po214_th}", flush=True)
    print("=" * 70, flush=True)

    sigma_vals_210 = [0.16, 0.18, 0.20, 0.22, 0.25]
    tau_vals_210 = [0.02, 0.04, 0.06, 0.08]
    p5_results = []

    for sh in sigma_vals_210:
        for th in tau_vals_210:
            cfg = copy.deepcopy(best_218_214_cfg)
            set_halo_params(cfg, "Po210", sigma_halo=sh, tau_halo=th)
            label = f"Po210 sh={sh:.2f} th={th:.2f}"
            m = run_pipeline(cfg, label)
            if m:
                m["phase"] = 5
                m["isotope"] = "Po210"
                m["f_halo"] = 0.08
                m["sigma_halo"] = sh
                m["tau_halo"] = th
                append_csv(m)
            p5_results.append(m)
            total_runs += 1

    best_p5 = pick_best(p5_results)
    if best_p5:
        best_po210_sh = best_p5["sigma_halo"]
        best_po210_th = best_p5["tau_halo"]
        print(f"\n  PHASE 5 BEST: sigma_halo={best_po210_sh}, tau_halo={best_po210_th} "
              f"(rb10={best_p5['rebin10_pull_sigma']:.3f})", flush=True)
    else:
        best_po210_sh = 0.20
        best_po210_th = 0.06
        print(f"\n  PHASE 5: No PASS results, using defaults", flush=True)

    # =========================================================
    # PHASE 6: Po210 f_halo sweep
    # =========================================================
    print("\n" + "=" * 70, flush=True)
    print(f"PHASE 6: Po210 f_halo sweep (sh={best_po210_sh}, th={best_po210_th})", flush=True)
    print("=" * 70, flush=True)

    fh_vals_210 = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    p6_results = []

    for fh in fh_vals_210:
        cfg = copy.deepcopy(best_218_214_cfg)
        set_halo_params(cfg, "Po210", f_halo=fh, sigma_halo=best_po210_sh, tau_halo=best_po210_th)
        label = f"Po210 fh={fh:.2f} sh={best_po210_sh:.2f} th={best_po210_th:.2f}"
        m = run_pipeline(cfg, label)
        if m:
            m["phase"] = 6
            m["isotope"] = "Po210"
            m["f_halo"] = fh
            m["sigma_halo"] = best_po210_sh
            m["tau_halo"] = best_po210_th
            append_csv(m)
        p6_results.append(m)
        total_runs += 1

    best_p6 = pick_best(p6_results)
    if best_p6:
        best_po210_fh = best_p6["f_halo"]
        print(f"\n  PHASE 6 BEST: f_halo={best_po210_fh} "
              f"(rb10={best_p6['rebin10_pull_sigma']:.3f})", flush=True)
    else:
        best_po210_fh = 0.08
        print(f"\n  PHASE 6: No PASS results, using default f_halo=0.08", flush=True)

    # =========================================================
    # PHASE 7: Combined verification
    # =========================================================
    print("\n" + "=" * 70, flush=True)
    print("PHASE 7: Combined verification with all optimal values", flush=True)
    print(f"  Po218: fh={best_po218_fh}, sh={best_po218_sh}, th={best_po218_th}", flush=True)
    print(f"  Po214: fh={best_po214_fh}, sh={best_po214_sh}, th={best_po214_th}", flush=True)
    print(f"  Po210: fh={best_po210_fh}, sh={best_po210_sh}, th={best_po210_th}", flush=True)
    print("=" * 70, flush=True)

    cfg = copy.deepcopy(BASE_CONFIG)
    set_halo_params(cfg, "Po218", f_halo=best_po218_fh, sigma_halo=best_po218_sh, tau_halo=best_po218_th)
    set_halo_params(cfg, "Po214", f_halo=best_po214_fh, sigma_halo=best_po214_sh, tau_halo=best_po214_th)
    set_halo_params(cfg, "Po210", f_halo=best_po210_fh, sigma_halo=best_po210_sh, tau_halo=best_po210_th)
    label = "COMBINED_OPTIMAL"
    m = run_pipeline(cfg, label)
    if m:
        m["phase"] = 7
        m["isotope"] = "ALL"
        m["f_halo"] = f"{best_po218_fh}/{best_po214_fh}/{best_po210_fh}"
        m["sigma_halo"] = f"{best_po218_sh}/{best_po214_sh}/{best_po210_sh}"
        m["tau_halo"] = f"{best_po218_th}/{best_po214_th}/{best_po210_th}"
        append_csv(m)
    total_runs += 1

    # =========================================================
    # PHASE 8: Fine-grained sweeps near Phase 1-6 optima
    # =========================================================
    elapsed_hours = (time.time() - t_start) / 3600
    print(f"\n  Elapsed: {elapsed_hours:.1f} hours, {total_runs} runs completed", flush=True)

    if elapsed_hours < 7:
        print("\n" + "=" * 70, flush=True)
        print("PHASE 8: Fine-grained Po218 sweep near optimum", flush=True)
        print("=" * 70, flush=True)

        # Fine grid around best Po218
        fine_sh = [best_po218_sh - 0.01, best_po218_sh, best_po218_sh + 0.01]
        fine_th = [best_po218_th - 0.01, best_po218_th, best_po218_th + 0.01]
        fine_fh = [best_po218_fh - 0.01, best_po218_fh, best_po218_fh + 0.01]

        # Filter out invalid values
        fine_sh = [x for x in fine_sh if x >= 0.18]
        fine_th = [x for x in fine_th if x >= 0.01]
        fine_fh = [x for x in fine_fh if x >= 0.02]

        for sh in fine_sh:
            for th in fine_th:
                for fh in fine_fh:
                    if sh == best_po218_sh and th == best_po218_th and fh == best_po218_fh:
                        continue  # skip already-tested combo
                    cfg = copy.deepcopy(best_218_214_cfg)
                    set_halo_params(cfg, "Po210", f_halo=best_po210_fh,
                                    sigma_halo=best_po210_sh, tau_halo=best_po210_th)
                    set_halo_params(cfg, "Po218", f_halo=fh, sigma_halo=sh, tau_halo=th)
                    label = f"P8_Po218 fh={fh:.3f} sh={sh:.3f} th={th:.3f}"
                    m = run_pipeline(cfg, label)
                    if m:
                        m["phase"] = 8
                        m["isotope"] = "Po218"
                        m["f_halo"] = fh
                        m["sigma_halo"] = sh
                        m["tau_halo"] = th
                        append_csv(m)
                    total_runs += 1
                    if (time.time() - t_start) / 3600 > 8:
                        break
                if (time.time() - t_start) / 3600 > 8:
                    break
            if (time.time() - t_start) / 3600 > 8:
                break

    # =========================================================
    # PHASE 9: Fine-grained Po214 sweep near optimum
    # =========================================================
    elapsed_hours = (time.time() - t_start) / 3600
    if elapsed_hours < 7.5:
        print("\n" + "=" * 70, flush=True)
        print("PHASE 9: Fine-grained Po214 sweep near optimum", flush=True)
        print("=" * 70, flush=True)

        fine_sh = [best_po214_sh - 0.01, best_po214_sh, best_po214_sh + 0.01]
        fine_th = [best_po214_th - 0.01, best_po214_th, best_po214_th + 0.01]
        fine_fh = [best_po214_fh - 0.01, best_po214_fh, best_po214_fh + 0.01]

        fine_sh = [x for x in fine_sh if x >= 0.20]
        fine_th = [x for x in fine_th if x >= 0.01]
        fine_fh = [x for x in fine_fh if x >= 0.02]

        for sh in fine_sh:
            for th in fine_th:
                for fh in fine_fh:
                    if sh == best_po214_sh and th == best_po214_th and fh == best_po214_fh:
                        continue
                    cfg = copy.deepcopy(BASE_CONFIG)
                    set_halo_params(cfg, "Po218", f_halo=best_po218_fh,
                                    sigma_halo=best_po218_sh, tau_halo=best_po218_th)
                    set_halo_params(cfg, "Po210", f_halo=best_po210_fh,
                                    sigma_halo=best_po210_sh, tau_halo=best_po210_th)
                    set_halo_params(cfg, "Po214", f_halo=fh, sigma_halo=sh, tau_halo=th)
                    label = f"P9_Po214 fh={fh:.3f} sh={sh:.3f} th={th:.3f}"
                    m = run_pipeline(cfg, label)
                    if m:
                        m["phase"] = 9
                        m["isotope"] = "Po214"
                        m["f_halo"] = fh
                        m["sigma_halo"] = sh
                        m["tau_halo"] = th
                        append_csv(m)
                    total_runs += 1
                    if (time.time() - t_start) / 3600 > 8.5:
                        break
                if (time.time() - t_start) / 3600 > 8.5:
                    break
            if (time.time() - t_start) / 3600 > 8.5:
                break

    # =========================================================
    # PHASE 10: Fine-grained Po210 sweep near optimum
    # =========================================================
    elapsed_hours = (time.time() - t_start) / 3600
    if elapsed_hours < 8:
        print("\n" + "=" * 70, flush=True)
        print("PHASE 10: Fine-grained Po210 sweep near optimum", flush=True)
        print("=" * 70, flush=True)

        fine_sh = [best_po210_sh - 0.01, best_po210_sh, best_po210_sh + 0.01]
        fine_th = [best_po210_th - 0.01, best_po210_th, best_po210_th + 0.01]
        fine_fh = [best_po210_fh - 0.01, best_po210_fh, best_po210_fh + 0.01]

        fine_sh = [x for x in fine_sh if x >= 0.16]
        fine_th = [x for x in fine_th if x >= 0.01]
        fine_fh = [x for x in fine_fh if x >= 0.02]

        for sh in fine_sh:
            for th in fine_th:
                for fh in fine_fh:
                    if sh == best_po210_sh and th == best_po210_th and fh == best_po210_fh:
                        continue
                    cfg = copy.deepcopy(BASE_CONFIG)
                    set_halo_params(cfg, "Po218", f_halo=best_po218_fh,
                                    sigma_halo=best_po218_sh, tau_halo=best_po218_th)
                    set_halo_params(cfg, "Po214", f_halo=best_po214_fh,
                                    sigma_halo=best_po214_sh, tau_halo=best_po214_th)
                    set_halo_params(cfg, "Po210", f_halo=fh, sigma_halo=sh, tau_halo=th)
                    label = f"P10_Po210 fh={fh:.3f} sh={sh:.3f} th={th:.3f}"
                    m = run_pipeline(cfg, label)
                    if m:
                        m["phase"] = 10
                        m["isotope"] = "Po210"
                        m["f_halo"] = fh
                        m["sigma_halo"] = sh
                        m["tau_halo"] = th
                        append_csv(m)
                    total_runs += 1
                    if (time.time() - t_start) / 3600 > 8.5:
                        break
                if (time.time() - t_start) / 3600 > 8.5:
                    break
            if (time.time() - t_start) / 3600 > 8.5:
                break

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    total_hours = (time.time() - t_start) / 3600
    print("\n" + "=" * 70, flush=True)
    print(f"SWEEP COMPLETE: {total_runs} runs in {total_hours:.1f} hours", flush=True)
    print("=" * 70, flush=True)
    print(f"\nRECOMMENDED CONFIG:", flush=True)
    print(f"  Po218: f_halo={best_po218_fh}, sigma_halo={best_po218_sh}, tau_halo={best_po218_th}", flush=True)
    print(f"  Po214: f_halo={best_po214_fh}, sigma_halo={best_po214_sh}, tau_halo={best_po214_th}", flush=True)
    print(f"  Po210: f_halo={best_po210_fh}, sigma_halo={best_po210_sh}, tau_halo={best_po210_th}", flush=True)
    print(f"\nFull results in: {CSV_PATH}", flush=True)

    # Clean up temp config
    tmp_config = os.path.join(BASE_DIR, "_sweep_halo_config.yaml")
    if os.path.exists(tmp_config):
        os.remove(tmp_config)


if __name__ == "__main__":
    main()
