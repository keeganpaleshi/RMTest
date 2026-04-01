"""Launch overnight scan series investigating I12p/I12r issues.

Based on findings:
- I12p (exp_unit + halos + EMG-Unk1): chi2/ndf=15.57 but Po218 collapsed, Unknown1 tail too broad
- I12r (exp_unit + NO halos + shelf-only + Unk1 Gaussian): chi2/ndf=130.48, Po218 fixed but 8-12 MeV region under-fit

Key insight: halos are symmetric (add counts on BOTH sides), but physics says
degraded alphas only LOSE energy (left-side only). However, completely removing
halos made the fit much worse because the 8-12 MeV continuum needs some
broadening mechanism. The solution space to explore:
- Shelf-only with wider/more flexible shelves
- Halos with very tight constraints (small f_halo, controlled sigma)
- Mixed: shelf for low-E tails + small constrained halos
- Po218 tau always tightened to prevent collapse

Waves of 5 parallel scans, run in series.
~7 min per scan, 8 waves = ~1 hour total.
"""
import subprocess
import sys
import os
import time
import yaml
import copy

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
TEMPLATE_PATH = os.path.join(BASE, "scan_configs", "I12p_config.yaml")

with open(TEMPLATE_PATH) as f:
    TEMPLATE = yaml.safe_load(f)


def make_config(scan_id, modifications, description=""):
    """Create a config from I12p template with targeted modifications."""
    cfg = copy.deepcopy(TEMPLATE)

    for key_path, value in modifications.items():
        parts = key_path.split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    cfg_path = os.path.join(BASE, "scan_configs", f"{scan_id}_config.yaml")
    with open(cfg_path, "w") as f:
        f.write(f"# {scan_id}: {description}\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return cfg_path


def launch_scan(scan_id, config_path):
    """Launch a single scan as detached process."""
    log_path = os.path.join(BASE, "results", f"{scan_id}_analyze.log")
    output_dir = os.path.join(BASE, "results", f"scan_{scan_id}")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        PYTHON, os.path.join(BASE, "analyze.py"),
        "--config", config_path,
        "--output-dir", output_dir,
    ]
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    with open(log_path, "w") as log_file:
        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=BASE,
            creationflags=flags,
        )
    return p.pid, log_path


def wait_for_completion(scan_ids, timeout_s=1800):
    """Wait for all scans to complete by monitoring log files."""
    start = time.time()
    pending = set(scan_ids)

    while pending and (time.time() - start) < timeout_s:
        time.sleep(15)
        for sid in list(pending):
            log_path = os.path.join(BASE, "results", f"{sid}_analyze.log")
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    content = f.read()
                if "Analysis complete" in content or "Error" in content or "Traceback" in content:
                    pending.discard(sid)
                    status = "DONE" if "Analysis complete" in content else "ERROR"
                    elapsed = time.time() - start
                    print(f"  {sid}: {status} ({elapsed:.0f}s)")

    if pending:
        print(f"  WARNING: {pending} did not complete within {timeout_s}s")
    return len(pending) == 0


def extract_result(scan_id):
    """Extract chi2/ndf and key params from a completed scan."""
    import json
    import glob
    pattern = os.path.join(BASE, f"results/scan_{scan_id}/*/summary.json")
    matches = glob.glob(pattern)
    if not matches:
        return None
    path = sorted(matches)[-1]
    with open(path) as f:
        d = json.load(f)
    sf = d.get("spectral_fit", {})
    return {
        "chi2_ndf": sf.get("chi2_ndf"),
        "ndf": sf.get("ndf"),
        "tau_Po218": sf.get("tau_Po218"),
        "f_shelf_Po218": sf.get("f_shelf_Po218"),
        "f_halo_Po218": sf.get("f_halo_Po218"),
        "S_Unknown1": sf.get("S_Unknown1"),
    }


# ═══════════════════════════════════════════════════════════════════
# Common mods: Po218 tau fix + Unknown1 plain Gaussian (always applied)
# ═══════════════════════════════════════════════════════════════════
COMMON = {
    "spectral_fit.tau_po218_prior_mean": 0.08,
    "spectral_fit.tau_po218_prior_sigma": 0.03,
    # Unknown1 as plain Gaussian (remove EMG/shelf/halo from template)
    "spectral_fit.extra_peaks.Unknown1.energy": 15.8,
    "spectral_fit.extra_peaks.Unknown1.sigma": [0.5, 0.1],
    "spectral_fit.extra_peaks.Unknown1.amplitude": [2000, 3000],
    "spectral_fit.extra_peaks.Unknown1.mu_sigma": 0.3,
    "spectral_fit.extra_peaks.Unknown1.mu_bounds": [15.0, 16.5],
}

def _strip_emg_unk1(mods):
    """Remove EMG/shelf/halo keys from Unknown1."""
    # The template has these — we need to explicitly remove them
    for k in list(mods.keys()):
        if "Unknown1" in k and any(x in k for x in ["use_emg", "tau.", "use_shelf", "f_shelf", "sigma_shelf", "use_halo", "f_halo", "sigma_halo", "tau_halo"]):
            del mods[k]
    return mods

# We'll handle Unknown1 cleanup by modifying the template directly
# after make_config runs. Easier to just delete the extra keys.

def make_config_clean(scan_id, modifications, description=""):
    """Create config, then clean up Unknown1 to be plain Gaussian."""
    cfg = copy.deepcopy(TEMPLATE)

    # First clean Unknown1 — remove EMG/shelf/halo from template
    unk1 = cfg.get("spectral_fit", {}).get("extra_peaks", {}).get("Unknown1", {})
    for k in ["use_emg", "tau", "use_shelf", "f_shelf", "sigma_shelf",
              "use_halo", "f_halo", "sigma_halo", "tau_halo"]:
        unk1.pop(k, None)

    # Apply modifications
    for key_path, value in modifications.items():
        parts = key_path.split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    cfg_path = os.path.join(BASE, "scan_configs", f"{scan_id}_config.yaml")
    with open(cfg_path, "w") as f:
        f.write(f"# {scan_id}: {description}\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return cfg_path


# ═══════════════════════════════════════════════════════════════════
# WAVE 1: Halo constraint exploration
# Keep halos but constrain them tightly. The 8-12 MeV continuum
# needs *some* broadening but it shouldn't extend to the right.
# ═══════════════════════════════════════════════════════════════════
wave1 = {
    "I13a": {
        "mods": {
            **COMMON,
            # Very small halos — f_halo capped at 0.10 (was 0.65)
            "spectral_fit.max_f_halo": 0.10,
            "spectral_fit.max_f_shelf": 0.50,
        },
        "desc": "exp + max_f_halo=0.10 (very tight) + Po218 fix + Unk1 Gaussian",
    },
    "I13b": {
        "mods": {
            **COMMON,
            # Small halos — f_halo capped at 0.20
            "spectral_fit.max_f_halo": 0.20,
        },
        "desc": "exp + max_f_halo=0.20 + Po218 fix + Unk1 Gaussian",
    },
    "I13c": {
        "mods": {
            **COMMON,
            # Moderate halos — f_halo capped at 0.30
            "spectral_fit.max_f_halo": 0.30,
        },
        "desc": "exp + max_f_halo=0.30 + Po218 fix + Unk1 Gaussian",
    },
    "I13d": {
        "mods": {
            **COMMON,
            # Small halos + tight sigma_halo (max 4x core, was 15x)
            "spectral_fit.max_f_halo": 0.20,
            "spectral_fit.sigma_halo_max_mult": 4.0,
        },
        "desc": "exp + f_halo≤0.20 + sigma_halo≤4x + Po218 fix + Unk1 Gaussian",
    },
    "I13e": {
        "mods": {
            **COMMON,
            # Small halos + tight sigma + fix Po218 halo
            "spectral_fit.max_f_halo": 0.15,
            "spectral_fit.sigma_halo_max_mult": 5.0,
            "spectral_fit.flags.fix_f_halo_Po218": True,
            "spectral_fit.flags.fix_sigma_halo_Po218": True,
            "spectral_fit.flags.fix_tau_halo_Po218": True,
        },
        "desc": "exp + f_halo≤0.15 + sigma_halo≤5x + Po218 halo FIXED + Unk1 Gaussian",
    },
}

# ═══════════════════════════════════════════════════════════════════
# WAVE 2: Shelf-only with wider/more flexible shelves
# Can the shelf alone explain the 8-12 MeV region if we give it
# more freedom?
# ═══════════════════════════════════════════════════════════════════
wave2 = {
    "I13f": {
        "mods": {
            **COMMON,
            # No halos, wider shelf range (8 MeV below peak)
            "spectral_fit.use_halo": {"Po210": False, "Po218": False, "Po214": False, "Po216": False, "Po212": False},
            "spectral_fit.shelf_range": 8.0,
            "spectral_fit.max_f_shelf": 0.60,
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + NO halos + shelf_range=8.0 + f_shelf≤0.60 + Unk1 Gaussian",
    },
    "I13g": {
        "mods": {
            **COMMON,
            # No halos, wider shelves, lower sigma_shelf floor
            "spectral_fit.use_halo": {"Po210": False, "Po218": False, "Po214": False, "Po216": False, "Po212": False},
            "spectral_fit.shelf_range": 5.0,
            "spectral_fit.max_f_shelf": 0.50,
            "spectral_fit.sigma_shelf_min": 0.01,
            # Raise shelf priors to be bigger
            "spectral_fit.f_shelf_Po210_prior": [0.15, 0.15],
            "spectral_fit.f_shelf_Po214_prior": [0.20, 0.15],
            "spectral_fit.f_shelf_Po218_prior": [0.20, 0.15],
            "spectral_fit.sigma_shelf_Po210_prior": [0.30, 0.15],
            "spectral_fit.sigma_shelf_Po214_prior": [0.30, 0.15],
            "spectral_fit.sigma_shelf_Po218_prior": [0.30, 0.15],
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + NO halos + shelf_range=5 + raised shelf priors + sigma_shelf_min=0.01",
    },
    "I13h": {
        "mods": {
            **COMMON,
            # No halos, max shelf fraction high, wider sigma priors
            "spectral_fit.use_halo": {"Po210": False, "Po218": False, "Po214": False, "Po216": False, "Po212": False},
            "spectral_fit.shelf_range": 5.0,
            "spectral_fit.max_f_shelf": 0.70,
            "spectral_fit.f_shelf_Po210_prior": [0.25, 0.20],
            "spectral_fit.f_shelf_Po214_prior": [0.30, 0.20],
            "spectral_fit.f_shelf_Po218_prior": [0.25, 0.15],
            "spectral_fit.f_shelf_Po212_prior": [0.40, 0.15],
            "spectral_fit.sigma_shelf_Po210_prior": [0.40, 0.20],
            "spectral_fit.sigma_shelf_Po214_prior": [0.40, 0.20],
            "spectral_fit.sigma_shelf_Po218_prior": [0.30, 0.15],
            "spectral_fit.sigma_shelf_Po212_prior": [0.50, 0.20],
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + NO halos + f_shelf≤0.70 + large shelf priors + Unk1 Gaussian",
    },
    "I13i": {
        "mods": {
            **COMMON,
            # Shelf for physics peaks + very small halo for Po214 only (largest peak)
            "spectral_fit.use_halo": {"Po210": False, "Po218": False, "Po214": True, "Po216": False, "Po212": False},
            "spectral_fit.max_f_halo": 0.15,
            "spectral_fit.sigma_halo_max_mult": 5.0,
            "spectral_fit.f_halo_Po214_prior": [0.10, 0.05],
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + halo ONLY on Po214 (small) + shelf on all + Unk1 Gaussian",
    },
    "I13j": {
        "mods": {
            **COMMON,
            # Shelf for physics peaks + small halo for Po210+Po214+Po212
            "spectral_fit.use_halo": {"Po210": True, "Po218": False, "Po214": True, "Po216": False, "Po212": True},
            "spectral_fit.max_f_halo": 0.15,
            "spectral_fit.sigma_halo_max_mult": 5.0,
            "spectral_fit.f_halo_Po210_prior": [0.05, 0.05],
            "spectral_fit.f_halo_Po214_prior": [0.10, 0.05],
            "spectral_fit.f_halo_Po212_prior": [0.08, 0.05],
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + small halo Po210/Po214/Po212 + shelf on all + no Po218 halo + Unk1 Gaussian",
    },
}

# ═══════════════════════════════════════════════════════════════════
# WAVE 3: Po218 specific — even with halos disabled on Po218, explore
# how to properly model Po218's degraded-alpha contribution
# ═══════════════════════════════════════════════════════════════════
wave3 = {
    "I13k": {
        "mods": {
            **COMMON,
            # Halos on all EXCEPT Po218, moderate caps
            "spectral_fit.use_halo": {"Po210": True, "Po218": False, "Po214": True, "Po216": False, "Po212": True},
            "spectral_fit.max_f_halo": 0.25,
            "spectral_fit.sigma_halo_max_mult": 6.0,
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + halo on Po210/214/212 (moderate) + no Po218 halo + Unk1 Gaussian",
    },
    "I13l": {
        "mods": {
            **COMMON,
            # Po218 tau even stronger to force proper tail
            "spectral_fit.tau_po218_prior_mean": 0.10,
            "spectral_fit.tau_po218_prior_sigma": 0.02,
            "spectral_fit.use_halo": {"Po210": True, "Po218": False, "Po214": True, "Po216": False, "Po212": True},
            "spectral_fit.max_f_halo": 0.25,
            "spectral_fit.sigma_halo_max_mult": 6.0,
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + strong Po218 tau (0.10±0.02) + halo on 210/214/212 + Unk1 Gaussian",
    },
    "I13m": {
        "mods": {
            **COMMON,
            # Same as I13k but with larger Po218 shelf
            "spectral_fit.use_halo": {"Po210": True, "Po218": False, "Po214": True, "Po216": False, "Po212": True},
            "spectral_fit.max_f_halo": 0.25,
            "spectral_fit.f_shelf_Po218_prior": [0.20, 0.10],
            "spectral_fit.sigma_shelf_Po218_prior": [0.20, 0.10],
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + halo 210/214/212 + larger Po218 shelf prior + Unk1 Gaussian",
    },
    "I13n": {
        "mods": {
            **COMMON,
            # All halos with moderate caps + Po218 halo fixed at small value
            "spectral_fit.max_f_halo": 0.25,
            "spectral_fit.sigma_halo_max_mult": 6.0,
            "spectral_fit.f_halo_Po218_prior": [0.05, 0.02],
            "spectral_fit.flags.fix_f_halo_Po218": True,
            "spectral_fit.flags.fix_sigma_halo_Po218": True,
            "spectral_fit.flags.fix_tau_halo_Po218": True,
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + all halos (f≤0.25) + Po218 halo FIXED at 0.05 + Unk1 Gaussian",
    },
    "I13o": {
        "mods": {
            **COMMON,
            # All halos very small + all shelves free + sigma_shelf_min lowered
            "spectral_fit.max_f_halo": 0.15,
            "spectral_fit.sigma_halo_max_mult": 4.0,
            "spectral_fit.sigma_shelf_min": 0.01,
            "spectral_fit.flags.fix_f_halo_Po218": True,
            "spectral_fit.flags.fix_sigma_halo_Po218": True,
            "spectral_fit.flags.fix_tau_halo_Po218": True,
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + tiny halos (f≤0.15, σ≤4x) + Po218 halo FIXED + σ_shelf_min=0.01 + Unk1 Gaussian",
    },
}

# ═══════════════════════════════════════════════════════════════════
# WAVE 4: Shelf range + combined best variations
# ═══════════════════════════════════════════════════════════════════
wave4 = {
    "I13p": {
        "mods": {
            **COMMON,
            # shelf_range=3 + moderate halo
            "spectral_fit.shelf_range": 3.0,
            "spectral_fit.use_halo": {"Po210": True, "Po218": False, "Po214": True, "Po216": False, "Po212": True},
            "spectral_fit.max_f_halo": 0.20,
            "spectral_fit.sigma_halo_max_mult": 5.0,
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + shelf_range=3 + halo 210/214/212 (f≤0.20) + Unk1 Gaussian",
    },
    "I13q": {
        "mods": {
            **COMMON,
            # shelf_range=2 + moderate halo
            "spectral_fit.shelf_range": 2.0,
            "spectral_fit.use_halo": {"Po210": True, "Po218": False, "Po214": True, "Po216": False, "Po212": True},
            "spectral_fit.max_f_halo": 0.20,
            "spectral_fit.sigma_halo_max_mult": 5.0,
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + shelf_range=2 + halo 210/214/212 (f≤0.20) + Unk1 Gaussian",
    },
    "I13r": {
        "mods": {
            **COMMON,
            # Best guess: moderate halos (not Po218), wider shelves, relaxed constraints
            "spectral_fit.use_halo": {"Po210": True, "Po218": False, "Po214": True, "Po216": False, "Po212": True},
            "spectral_fit.max_f_halo": 0.20,
            "spectral_fit.sigma_halo_max_mult": 5.0,
            "spectral_fit.sigma_shelf_min": 0.01,
            "spectral_fit.tau_halo_max_mult": 8.0,
            "spectral_fit.f_shelf_Po218_prior": [0.15, 0.10],
            "spectral_fit.sigma_shelf_Po218_prior": [0.20, 0.10],
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + halo 210/214/212 (f≤0.20) + relaxed constraints + Po218 shelf tighter",
    },
    "I13s": {
        "mods": {
            **COMMON,
            # Same as I13r but also fix halo sigma/tau at priors (only float f_halo)
            "spectral_fit.use_halo": {"Po210": True, "Po218": False, "Po214": True, "Po216": False, "Po212": True},
            "spectral_fit.max_f_halo": 0.20,
            "spectral_fit.flags.fix_sigma_halo_Po210": True,
            "spectral_fit.flags.fix_tau_halo_Po210": True,
            "spectral_fit.flags.fix_sigma_halo_Po214": True,
            "spectral_fit.flags.fix_tau_halo_Po214": True,
            "spectral_fit.flags.fix_sigma_halo_Po212": True,
            "spectral_fit.flags.fix_tau_halo_Po212": True,
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + halo 210/214/212 (f free, σ/τ FIXED) + all shelves free + Unk1 Gaussian",
    },
    "I13t": {
        "mods": {
            **COMMON,
            # No halos at all, but very large shelf fractions allowed
            "spectral_fit.use_halo": {"Po210": False, "Po218": False, "Po214": False, "Po216": False, "Po212": False},
            "spectral_fit.max_f_shelf": 0.80,
            "spectral_fit.shelf_range": 5.0,
            "spectral_fit.sigma_shelf_min": 0.01,
            "spectral_fit.f_shelf_Po210_prior": [0.30, 0.25],
            "spectral_fit.f_shelf_Po214_prior": [0.35, 0.20],
            "spectral_fit.f_shelf_Po218_prior": [0.25, 0.15],
            "spectral_fit.f_shelf_Po212_prior": [0.45, 0.20],
            "spectral_fit.sigma_shelf_Po210_prior": [0.50, 0.25],
            "spectral_fit.sigma_shelf_Po214_prior": [0.50, 0.25],
            "spectral_fit.sigma_shelf_Po218_prior": [0.30, 0.15],
            "spectral_fit.sigma_shelf_Po212_prior": [0.60, 0.25],
            "spectral_fit.flags.fix_f_shelf_Po210": False,
            "spectral_fit.flags.fix_sigma_shelf_Po210": False,
            "spectral_fit.flags.fix_f_shelf_Po218": False,
            "spectral_fit.flags.fix_sigma_shelf_Po218": False,
            "spectral_fit.flags.fix_f_shelf_Po214": False,
            "spectral_fit.flags.fix_sigma_shelf_Po214": False,
            "spectral_fit.flags.fix_f_shelf_Po212": False,
            "spectral_fit.flags.fix_sigma_shelf_Po212": False,
        },
        "desc": "exp + NO halos + max_f_shelf=0.80 + very large shelf priors + Unk1 Gaussian",
    },
}

all_waves = [
    ("Wave 1: Halo constraint tightening", wave1),
    ("Wave 2: Shelf-only with wider shelves", wave2),
    ("Wave 3: Po218 specific + halo configurations", wave3),
    ("Wave 4: Shelf range + combined best", wave4),
]

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("OVERNIGHT SCAN SERIES")
    print(f"Total scans: {sum(len(w) for _, w in all_waves)}")
    print(f"Waves: {len(all_waves)} (5 scans each, ~7 min/scan)")
    print(f"Estimated total time: {len(all_waves) * 8} minutes")
    print("=" * 70)

    results_table = []

    for wave_name, wave_scans in all_waves:
        print(f"\n{'─' * 60}")
        print(f"Launching {wave_name} ({len(wave_scans)} scans)")
        print(f"{'─' * 60}")

        scan_ids = []
        for scan_id, spec in wave_scans.items():
            cfg_path = make_config_clean(scan_id, spec["mods"], spec["desc"])
            pid, log_path = launch_scan(scan_id, cfg_path)
            print(f"  Launched {scan_id}: PID={pid} — {spec['desc']}")
            scan_ids.append(scan_id)

        print(f"\n  Waiting for {wave_name} to complete...")
        ok = wait_for_completion(scan_ids, timeout_s=1800)

        # Extract results
        for scan_id in scan_ids:
            r = extract_result(scan_id)
            desc = wave_scans[scan_id]["desc"]
            if r and r["chi2_ndf"] is not None:
                results_table.append((scan_id, r["chi2_ndf"], r["ndf"], desc))
                print(f"  {scan_id}: chi2/ndf={r['chi2_ndf']:.2f} — {desc}")
            else:
                print(f"  {scan_id}: FAILED — {desc}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Scan':8s} {'chi2/ndf':>10s} {'ndf':>5s}  Description")
    print("-" * 70)
    for scan_id, chi2, ndf, desc in sorted(results_table, key=lambda x: x[1]):
        print(f"{scan_id:8s} {chi2:10.2f} {ndf:5d}  {desc}")
    print("=" * 70)
