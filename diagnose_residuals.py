#!/usr/bin/env python3
"""
Diagnostic script: run the analysis pipeline and intercept the spectral fit
to produce detailed residual diagnostics.
"""
import sys, json, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

logging.basicConfig(level=logging.INFO)

# ── Use the actual pipeline to load & calibrate ─────────────────────────
from io_utils import load_config, load_events, apply_burst_filter
from calibration import derive_calibration_constants, apply_calibration
from fitting import fit_spectrum, FitResult

cfg = load_config('config.yaml')
csv_path = cfg.get('input', cfg.get('input_csv', 'merged_output.csv'))
if isinstance(csv_path, dict):
    csv_path = csv_path.get('path', 'merged_output.csv')
df = load_events(csv_path, column_map=cfg.get('columns'))

# Apply same cuts as analyze.py
noise_cut = cfg.get('noise_cut_adc', 0)
if isinstance(noise_cut, dict):
    noise_cut = noise_cut.get('threshold', 100)
df = df[df['adc'] > noise_cut]
df, _ = apply_burst_filter(df, cfg)

# Calibrate
cal = derive_calibration_constants(df['adc'].values, cfg)
# CalibrationResult uses polynomial coeffs and predict()
df['energy_MeV'] = cal.predict(df['adc'].values)
# Extract slope/intercept for bin-edge construction
coeffs = cal.coeffs
intercept = coeffs[0]
slope = coeffs[1] if len(coeffs) > 1 else 0.004299

E_all = df['energy_MeV'].values
print(f"Calibration: slope={slope:.6f}, intercept={intercept:.4f}")
print(f"Events after cuts: {len(E_all)}")

# ── Spectral fit config ─────────────────────────────────────────────────
spec_cfg = cfg.get('spectral_fit', {})
fit_range = spec_cfg.get('fit_energy_range', [1.7, 8.3])

# Build bin edges (1 ADC channel per bin, same as analyze.py)
binning_mode = spec_cfg.get('spectral_binning_mode', 'adc')
if binning_mode == 'adc':
    adc_bin_width = spec_cfg.get('adc_bin_width', 1)
    adc_min = int(np.floor((fit_range[0] - intercept) / slope))
    adc_max = int(np.ceil((fit_range[1] - intercept) / slope))
    bin_edges_adc = np.arange(adc_min, adc_max + 2, adc_bin_width) - 0.5
    bin_edges = bin_edges_adc * slope + intercept
else:
    ebin_w = spec_cfg.get('energy_bin_width', 0.013)
    bin_edges = np.arange(fit_range[0], fit_range[1] + ebin_w, ebin_w)

hist, _ = np.histogram(E_all, bins=bin_edges)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dE = np.diff(bin_edges)
print(f"Bins: {len(hist)}, bin width ~{np.mean(dE)*1000:.1f} keV")

# ── Use parameters from the summary JSON ────────────────────────────────
with open('results/20260309T235927Z/summary.json') as f:
    summary = json.load(f)
params = summary['spectral_fit']

print(f"\n=== FIT RESULTS ===")
print(f"chi2/NDF = {params.get('chi2', 0):.1f}/{params.get('ndf', 0)} = {params.get('chi2_ndf', 0):.2f}")
for iso in ['Po210', 'Po218', 'Po214']:
    print(f"{iso}: mu={params[f'mu_{iso}']:.4f}, S={params[f'S_{iso}']:.0f}, "
          f"tau={params.get(f'tau_{iso}', 0):.5f}, sigma={params.get(f'sigma_{iso}', params.get('sigma0', 0)):.4f}, "
          f"f_shelf={params.get(f'f_shelf_{iso}', 'N/A')}")
print(f"Background: b0={params['b0']:.4f}, b1={params['b1']:.4f}, b2={params.get('b2', 'N/A')}, S_bkg={params['S_bkg']:.0f}")
print(f"sigma0={params['sigma0']:.4f}")

# ── Build model prediction using the actual spectral intensity function ──
sys.path.insert(0, 'src')
from rmtest.spectral.intensity import build_spectral_intensity
from constants import safe_exp

sigma0 = params['sigma0']

use_emg_map = {iso: spec_cfg.get('use_emg', {}).get(iso, True) for iso in ['Po210', 'Po218', 'Po214']}
use_shelf_map = {iso: spec_cfg.get('use_shelf', {}).get(iso, False) for iso in ['Po210', 'Po218', 'Po214']}
use_halo_map = {iso: spec_cfg.get('use_halo', {}).get(iso, False) for iso in ['Po210', 'Po218', 'Po214']}
bkg_model = cfg.get('analysis', {}).get('background_model', 'loglin_unit')

intensity_fn = build_spectral_intensity(
    ['Po210', 'Po218', 'Po214'],
    use_emg_map,
    tuple(fit_range),
    background_model=bkg_model,
    use_shelf=use_shelf_map,
    use_halo=use_halo_map,
)

# Build the full parameter dict for the model
model_params = dict(params)
# Ensure N_ keys exist (model uses N_ internally)
for iso in ['Po210', 'Po218', 'Po214']:
    model_params[f'N_{iso}'] = params[f'S_{iso}']

# Apply DNL correction if available in the summary
dnl_factors = np.ones_like(dE)
dnl_info = params.get('_dnl', {})
if dnl_info.get('dnl_applied', False):
    dnl_raw = np.array(dnl_info['dnl_factors'])
    n_fit = len(dnl_raw)
    n_diag = len(dE)
    if n_fit == n_diag:
        dnl_factors = dnl_raw
    elif abs(n_fit - n_diag) <= 5:
        # Small mismatch: truncate/pad to match, using 1.0 for edges
        n_common = min(n_fit, n_diag)
        # Center-align: offset to match peak positions
        offset = (n_diag - n_fit) // 2
        for i in range(n_common):
            src = i
            dst = i + max(offset, 0)
            if 0 <= dst < n_diag and 0 <= src < n_fit:
                dnl_factors[dst] = dnl_raw[src]
    else:
        print(f"WARNING: DNL array length {n_fit} vs {n_diag} bins, "
              f"skipping DNL correction")
    print(f"DNL correction applied: {n_fit} factors -> {n_diag} bins, "
          f"range [{dnl_factors.min():.4f}, {dnl_factors.max():.4f}]")
else:
    print("No DNL correction in summary")

# Total model density at bin centers, then convert to counts per bin
total_density = intensity_fn(bin_centers, model_params, tuple(fit_range))
model_total = total_density * dE * dnl_factors

# Per-isotope components for plotting (suppress each isotope one at a time)
model_components = {}
for iso in ['Po210', 'Po218', 'Po214']:
    # Zero out this isotope to get everything-else
    params_without = dict(model_params)
    params_without[f'N_{iso}'] = 0.0
    params_without[f'S_{iso}'] = 0.0
    density_without = intensity_fn(bin_centers, params_without, tuple(fit_range))
    model_components[iso] = (total_density - density_without) * dE * dnl_factors

# Background (zero out all peaks)
params_bkg_only = dict(model_params)
for iso in ['Po210', 'Po218', 'Po214']:
    params_bkg_only[f'N_{iso}'] = 0.0
    params_bkg_only[f'S_{iso}'] = 0.0
model_components['bkg'] = intensity_fn(bin_centers, params_bkg_only, tuple(fit_range)) * dE * dnl_factors

residuals = hist - model_total

# Also compute model WITHOUT DNL to compare
model_no_dnl = total_density * dE
residuals_no_dnl = hist - model_no_dnl

# Pull = residual / sqrt(expected) for Poisson
with np.errstate(divide='ignore', invalid='ignore'):
    pull = np.where(model_total > 0, residuals / np.sqrt(np.maximum(model_total, 1)), 0)

mask_fit = (bin_centers >= fit_range[0]) & (bin_centers <= fit_range[1])
mask_stats = mask_fit & (model_total > 5)

pull_valid = pull[mask_stats]
chi2_pull = np.sum(pull[mask_fit]**2)
ndf = np.sum(mask_fit) - 14

# === DETAILED RAW RESIDUAL ANALYSIS ===
print(f"\n=== RAW RESIDUAL ANALYSIS ===")
print(f"Top 20 largest |residuals| (WITH DNL correction):")
idx_sorted = np.argsort(np.abs(residuals))[::-1]
for i in range(20):
    idx = idx_sorted[i]
    pct = 100*residuals[idx]/max(model_total[idx],1)
    print(f"  E={bin_centers[idx]:.4f} MeV: data={hist[idx]:>7d} model={model_total[idx]:>10.1f} "
          f"resid={residuals[idx]:>+8.0f} ({pct:>+6.1f}%) dnl={dnl_factors[idx]:.4f}")

print(f"\nTop 20 largest |residuals| (WITHOUT DNL):")
idx_sorted2 = np.argsort(np.abs(residuals_no_dnl))[::-1]
for i in range(20):
    idx = idx_sorted2[i]
    pct = 100*residuals_no_dnl[idx]/max(model_no_dnl[idx],1)
    print(f"  E={bin_centers[idx]:.4f} MeV: data={hist[idx]:>7d} model={model_no_dnl[idx]:>10.1f} "
          f"resid={residuals_no_dnl[idx]:>+8.0f} ({pct:>+6.1f}%)")

print(f"\n=== PEAK REGION RAW RESIDUAL SUMMARY ===")
for region_name, lo, hi in [('Po210 peak region', 4.8, 5.6), ('Po218 peak region', 5.7, 6.3),
                              ('Po214 peak region', 7.2, 8.1), ('Valley', 5.5, 5.8),
                              ('Low E background', 1.7, 4.0)]:
    mask = (bin_centers >= lo) & (bin_centers < hi)
    r = residuals[mask]
    r_nd = residuals_no_dnl[mask]
    d = hist[mask]
    m = model_total[mask]
    m_nd = model_no_dnl[mask]
    n = np.sum(mask)
    print(f"\n  {region_name} ({lo}-{hi} MeV, {n} bins):")
    print(f"    Total data: {np.sum(d):>12,.0f}")
    print(f"    Total model (DNL):   {np.sum(m):>12,.0f}  sum_resid={np.sum(r):>+10,.0f} ({100*np.sum(r)/max(np.sum(d),1):>+.2f}%)")
    print(f"    Total model (noDNL): {np.sum(m_nd):>12,.0f}  sum_resid={np.sum(r_nd):>+10,.0f} ({100*np.sum(r_nd)/max(np.sum(d),1):>+.2f}%)")
    print(f"    Max |resid| (DNL):   {np.max(np.abs(r)):>8,.0f}  at E={bin_centers[mask][np.argmax(np.abs(r))]:.4f}")
    print(f"    Max |resid| (noDNL): {np.max(np.abs(r_nd)):>8,.0f}  at E={bin_centers[mask][np.argmax(np.abs(r_nd))]:.4f}")
    print(f"    RMS resid (DNL):     {np.sqrt(np.mean(r**2)):>8,.0f}")
    print(f"    RMS resid (noDNL):   {np.sqrt(np.mean(r_nd**2)):>8,.0f}")
    # Check for coherent run of same-sign residuals
    pos_run = neg_run = max_pos_run = max_neg_run = 0
    for ri in r:
        if ri > 0:
            pos_run += 1; neg_run = 0; max_pos_run = max(max_pos_run, pos_run)
        else:
            neg_run += 1; pos_run = 0; max_neg_run = max(max_neg_run, neg_run)
    print(f"    Longest positive run: {max_pos_run} bins, longest negative run: {max_neg_run} bins")

print(f"\n=== RESIDUAL DIAGNOSTICS ===")
print(f"Chi2 from pulls: {chi2_pull:.0f}/{ndf} = {chi2_pull/ndf:.1f}")
print(f"Pull (model>5): mean={np.mean(pull_valid):.3f}, std={np.std(pull_valid):.3f}")
print(f"|pull|>2: {np.mean(np.abs(pull_valid)>2):.1%}")
print(f"|pull|>3: {np.mean(np.abs(pull_valid)>3):.1%}")
print(f"Max |pull|: {np.max(np.abs(pull_valid)):.1f}")

# ── Per-region analysis ─────────────────────────────────────────────────
print(f"\n=== REGIONAL BREAKDOWN ===")
regions = {
    'Low E (1.7-4.5)':       (1.7, 4.5),
    'Po210 tail (4.5-5.0)':  (4.5, 5.0),
    'Po210 peak (5.0-5.5)':  (5.0, 5.5),
    'Valley (5.5-5.8)':      (5.5, 5.8),
    'Po218 (5.8-6.3)':       (5.8, 6.3),
    'Mid (6.3-7.3)':         (6.3, 7.3),
    'Po214 (7.3-8.0)':       (7.3, 8.0),
    'High E (8.0-8.3)':      (8.0, 8.3),
}

region_names, region_chi2_ndf, region_mean_pull = [], [], []
for name, (lo, hi) in regions.items():
    rmask = (bin_centers >= lo) & (bin_centers < hi) & mask_fit
    n_bins = np.sum(rmask)
    if n_bins > 0:
        c2 = np.sum(pull[rmask]**2)
        mp = np.mean(pull[rmask])
        region_names.append(name)
        region_chi2_ndf.append(c2 / max(n_bins, 1))
        region_mean_pull.append(mp)
        print(f"  {name}: chi2/nbin = {c2:.0f}/{n_bins} = {c2/n_bins:.1f}, mean pull = {mp:.2f}")

# ── FIGURE 1: Detailed residual diagnostic ──────────────────────────────
OUTDIR = 'results/20260309T235927Z'

fig = plt.figure(figsize=(18, 22))
gs = GridSpec(6, 2, figure=fig, height_ratios=[3, 1.5, 1.5, 2, 2, 2])

# 1. Full spectrum (log scale)
ax = fig.add_subplot(gs[0, :])
ax.step(bin_centers[mask_fit], hist[mask_fit], where='mid', color='gray', alpha=0.6, label='Data', linewidth=0.5)
ax.plot(bin_centers[mask_fit], model_total[mask_fit], 'r-', linewidth=1, label='Total model')
for iso, color in [('Po210', 'green'), ('Po218', 'blue'), ('Po214', 'orange')]:
    ax.plot(bin_centers[mask_fit], model_components[iso][mask_fit], color=color, linewidth=0.8, label=iso, alpha=0.8)
ax.plot(bin_centers[mask_fit], model_components['bkg'][mask_fit], 'brown', linewidth=0.8, label='Background')
ax.set_ylabel('Counts per bin')
ax.set_title(f'Full Spectrum Fit  |  χ²/NDF = {params.get("chi2",0):.0f}/{params.get("ndf",0)} = {params.get("chi2_ndf",0):.2f}')
ax.legend(fontsize=9)
ax.set_xlim(fit_range)
ax.set_yscale('log')
ax.set_ylim(0.5, None)

# 2. Residuals
ax = fig.add_subplot(gs[1, :])
ax.bar(bin_centers[mask_fit], residuals[mask_fit], width=dE[mask_fit], color='steelblue', alpha=0.6, linewidth=0)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Data − Model')
ax.set_xlim(fit_range)
ax.set_title('Residuals')

# 3. Pull
ax = fig.add_subplot(gs[2, :])
colors_pull = np.where(np.abs(pull[mask_fit]) > 3, 'red', 'steelblue')
ax.bar(bin_centers[mask_fit], pull[mask_fit], width=dE[mask_fit], color=colors_pull, alpha=0.6, linewidth=0)
ax.axhline(0, color='k', linewidth=0.5)
ax.axhline(3, color='r', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axhline(-3, color='r', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_ylabel('Pull')
ax.set_xlim(fit_range)
ax.set_title('Standardized Pulls (|pull| > 3 in red)')

# 4. Zoomed: Po-210 region
ax = fig.add_subplot(gs[3, 0])
z = (bin_centers >= 4.5) & (bin_centers <= 5.7) & mask_fit
ax.step(bin_centers[z], hist[z], where='mid', color='gray', alpha=0.7, linewidth=0.8, label='Data')
ax.plot(bin_centers[z], model_total[z], 'r-', linewidth=1.2, label='Model')
ax.plot(bin_centers[z], model_components['Po210'][z], 'g-', linewidth=0.8, alpha=0.7, label='Po210')
ax.plot(bin_centers[z], model_components['bkg'][z], 'brown', linewidth=0.8, alpha=0.7, label='Bkg')
ax.set_title('Po-210 region (4.5–5.7 MeV)')
ax.legend(fontsize=8)
ax.set_ylabel('Counts')

ax_r = fig.add_subplot(gs[3, 1])
ax_r.bar(bin_centers[z], pull[z], width=dE[z], color='steelblue', alpha=0.6, linewidth=0)
ax_r.axhline(0, color='k', linewidth=0.5)
ax_r.axhline(3, color='r', linewidth=0.5, linestyle='--')
ax_r.axhline(-3, color='r', linewidth=0.5, linestyle='--')
ax_r.set_title('Pull: Po-210')
ax_r.set_ylabel('Pull')

# 5. Zoomed: Po-218 region
ax = fig.add_subplot(gs[4, 0])
z = (bin_centers >= 5.7) & (bin_centers <= 6.4) & mask_fit
ax.step(bin_centers[z], hist[z], where='mid', color='gray', alpha=0.7, linewidth=0.8, label='Data')
ax.plot(bin_centers[z], model_total[z], 'r-', linewidth=1.2, label='Model')
ax.plot(bin_centers[z], model_components['Po218'][z], 'b-', linewidth=0.8, alpha=0.7, label='Po218')
ax.plot(bin_centers[z], model_components['bkg'][z], 'brown', linewidth=0.8, alpha=0.7, label='Bkg')
ax.set_title('Po-218 region (5.7–6.4 MeV)')
ax.legend(fontsize=8)
ax.set_ylabel('Counts')

ax_r = fig.add_subplot(gs[4, 1])
ax_r.bar(bin_centers[z], pull[z], width=dE[z], color='steelblue', alpha=0.6, linewidth=0)
ax_r.axhline(0, color='k', linewidth=0.5)
ax_r.axhline(3, color='r', linewidth=0.5, linestyle='--')
ax_r.axhline(-3, color='r', linewidth=0.5, linestyle='--')
ax_r.set_title('Pull: Po-218')
ax_r.set_ylabel('Pull')

# 6. Zoomed: Po-214 region
ax = fig.add_subplot(gs[5, 0])
z = (bin_centers >= 7.0) & (bin_centers <= 8.3) & mask_fit
ax.step(bin_centers[z], hist[z], where='mid', color='gray', alpha=0.7, linewidth=0.8, label='Data')
ax.plot(bin_centers[z], model_total[z], 'r-', linewidth=1.2, label='Model')
ax.plot(bin_centers[z], model_components['Po214'][z], color='orange', linewidth=0.8, alpha=0.7, label='Po214')
ax.plot(bin_centers[z], model_components['bkg'][z], 'brown', linewidth=0.8, alpha=0.7, label='Bkg')
ax.set_title('Po-214 region (7.0–8.3 MeV)')
ax.legend(fontsize=8)
ax.set_ylabel('Counts')
ax.set_xlabel('Energy [MeV]')

ax_r = fig.add_subplot(gs[5, 1])
ax_r.bar(bin_centers[z], pull[z], width=dE[z], color='steelblue', alpha=0.6, linewidth=0)
ax_r.axhline(0, color='k', linewidth=0.5)
ax_r.axhline(3, color='r', linewidth=0.5, linestyle='--')
ax_r.axhline(-3, color='r', linewidth=0.5, linestyle='--')
ax_r.set_title('Pull: Po-214')
ax_r.set_ylabel('Pull')
ax_r.set_xlabel('Energy [MeV]')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/residual_diagnostic.png', dpi=150)
print(f"\nSaved {OUTDIR}/residual_diagnostic.png")

# ── FIGURE 2: Pull statistics & autocorrelation ─────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Pull histogram
ax = axes2[0, 0]
ax.hist(pull_valid, bins=80, density=True, alpha=0.7, color='steelblue',
        label=f'Pulls (μ={np.mean(pull_valid):.2f}, σ={np.std(pull_valid):.2f})')
x = np.linspace(-8, 8, 200)
ax.plot(x, norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
ax.set_title('Pull Distribution')
ax.legend()
ax.set_xlabel('Pull value')

# Running mean (systematic trends)
ax = axes2[0, 1]
E_s = bin_centers[mask_stats]
p_s = pull[mask_stats]
window = 30
if len(p_s) > window:
    rm = np.convolve(p_s, np.ones(window)/window, mode='valid')
    rE = np.convolve(E_s, np.ones(window)/window, mode='valid')
    ax.plot(rE, rm, 'b-', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(1, color='r', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(-1, color='r', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_title(f'Running Mean of Pull (window={window} bins)')
ax.set_xlabel('Energy [MeV]')
ax.set_ylabel('Mean pull')

# Regional chi2
ax = axes2[1, 0]
colors_bar = ['red' if v > 3 else 'orange' if v > 1.5 else 'green' for v in region_chi2_ndf]
ax.barh(range(len(region_names)), region_chi2_ndf, color=colors_bar, alpha=0.7)
ax.set_yticks(range(len(region_names)))
ax.set_yticklabels(region_names, fontsize=9)
ax.axvline(1, color='k', linewidth=1, linestyle='--', label='χ²/bin = 1 (ideal)')
ax.set_xlabel('χ²/bin')
ax.set_title('χ² per Bin by Region')
ax.legend()

# Autocorrelation
ax = axes2[1, 1]
p_cent = p_s - np.mean(p_s)
if len(p_cent) > 10:
    from numpy.fft import fft, ifft
    acf = np.real(ifft(np.abs(fft(p_cent))**2)) / (np.var(p_cent) * len(p_cent))
    max_lag = min(80, len(acf)//2)
    ax.bar(range(1, max_lag), acf[1:max_lag], color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(2/np.sqrt(len(p_cent)), color='r', linewidth=0.5, linestyle='--', label='95% CL')
    ax.axhline(-2/np.sqrt(len(p_cent)), color='r', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Lag (bins)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Pull Autocorrelation')
    ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTDIR}/residual_diagnostic2.png', dpi=150)
print(f"Saved {OUTDIR}/residual_diagnostic2.png")
