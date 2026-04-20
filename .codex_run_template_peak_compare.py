from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


WORKSPACE = Path(r"C:\Users\keega\Radon\RMTest")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def latest_timestamp_dir(path: Path) -> Path | None:
    dirs = sorted([p for p in path.iterdir() if p.is_dir()]) if path.exists() else []
    return dirs[-1] if dirs else None


def variant_definitions() -> dict[str, dict[str, Any]]:
    return {
        "baseline_passive": {},
        "po216_active_shape": {
            "spectral_fit": {"shared_shape_passive": ["Bi212"]},
        },
        "bi212_active_shape": {
            "spectral_fit": {"shared_shape_passive": ["Po216"]},
        },
        "both_active_shape": {
            "spectral_fit": {"shared_shape_passive": []},
        },
    }


def summarise_run(timestamp_dir: Path) -> dict[str, Any]:
    summary_path = timestamp_dir / "summary.json"
    diag_path = timestamp_dir / "template_per_bin_diagnostics.json"
    debug_path = timestamp_dir / "template_fitting_debug.txt"
    template_dir = timestamp_dir / "template_bin_fits"

    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    diagnostics: list[dict[str, Any]] = []
    if diag_path.exists():
        with diag_path.open("r", encoding="utf-8") as fh:
            diag_obj = json.load(fh)
        if isinstance(diag_obj, dict):
            diagnostics = diag_obj.get("per_bin_diagnostics", [])
        elif isinstance(diag_obj, list):
            diagnostics = diag_obj

    template_summary = summary.get("template_fitting", {})
    chi2_values = [
        float(row.get("chi2_ndf"))
        for row in diagnostics
        if isinstance(row, dict) and row.get("chi2_ndf") is not None
    ]
    suspicious_counts: dict[str, int] = {}
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        for name in row.get("suspicious_bound_hit_params", []):
            suspicious_counts[str(name)] = suspicious_counts.get(str(name), 0) + 1

    linear_plots = 0
    log_plots = 0
    if template_dir.exists():
        linear_plots = len(
            [p for p in template_dir.glob("template_fit_bin_*.png") if not p.name.endswith("_log.png")]
        )
        log_plots = len(list(template_dir.glob("*_log.png")))

    series_pngs = sorted(
        p.name
        for p in timestamp_dir.glob("time_series_*.png")
        if not p.name.endswith("_log.png")
    )

    return {
        "template_status": template_summary.get("status"),
        "n_fitted": template_summary.get("n_fitted"),
        "n_valid": template_summary.get("n_valid"),
        "n_plot_entries": template_summary.get("n_plot_entries"),
        "chi2_mean": template_summary.get("chi2_ndf_mean"),
        "chi2_median": template_summary.get("chi2_ndf_median"),
        "chi2_max": max(chi2_values) if chi2_values else None,
        "shift_rms_kev": template_summary.get("centroid_shift_rms_kev"),
        "shift_p95_kev": template_summary.get("p95_abs_shift_kev"),
        "shift_max_kev": template_summary.get("max_abs_shift_kev"),
        "bins_with_suspicious_bound_hits": template_summary.get("bins_with_suspicious_bound_hits"),
        "S_Po216_bound_hits": suspicious_counts.get("S_Po216", 0),
        "S_Bi212_bound_hits": suspicious_counts.get("S_Bi212", 0),
        "lambda_beta_Po214_bound_hits": suspicious_counts.get("lambda_beta_Po214", 0),
        "linear_plot_count": linear_plots,
        "log_plot_count": log_plots,
        "has_bi212_series_plot": any(name.startswith("time_series_Bi212") for name in series_pngs),
        "series_pngs": ",".join(series_pngs),
        "template_debug_exists": debug_path.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=".codex_run_merged_output_template_review_v3.yaml")
    parser.add_argument("--input", default="merged_output.csv")
    parser.add_argument("--output-prefix", default=".codex_run_template_peak_compare")
    parser.add_argument("--run-period-start", default="2024-11-01T00:00:00Z")
    parser.add_argument("--run-period-end", default="2024-12-01T00:00:00Z")
    parser.add_argument("--variants", default="baseline_passive,po216_active_shape,bi212_active_shape,both_active_shape")
    args = parser.parse_args()

    base_config_path = WORKSPACE / args.base_config
    base_config = load_yaml(base_config_path)
    base_config.setdefault("plotting", {})
    base_config["plotting"]["plot_template_bin_fits"] = True
    base_config["plotting"]["plot_template_bin_fits_bad_only"] = False
    base_config["plotting"]["plot_template_bin_fits_log_scale"] = True

    selected = [part.strip() for part in str(args.variants).split(",") if part.strip()]
    variants = variant_definitions()

    session_id = now_utc().strftime("%Y%m%dT%H%M%SZ")
    sweep_root = WORKSPACE / f"{args.output_prefix}_{session_id}"
    config_dir = sweep_root / "configs"
    output_root = sweep_root / "outputs"
    log_dir = sweep_root / "logs"
    summary_csv = sweep_root / "summary.csv"
    for path in (sweep_root, config_dir, output_root, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "variant",
                "status",
                "exit_code",
                "timestamp_dir",
                "template_status",
                "n_fitted",
                "n_valid",
                "n_plot_entries",
                "chi2_mean",
                "chi2_median",
                "chi2_max",
                "shift_rms_kev",
                "shift_p95_kev",
                "shift_max_kev",
                "bins_with_suspicious_bound_hits",
                "S_Po216_bound_hits",
                "S_Bi212_bound_hits",
                "lambda_beta_Po214_bound_hits",
                "linear_plot_count",
                "log_plot_count",
                "has_bi212_series_plot",
                "series_pngs",
            ]
        )

    for variant in selected:
        if variant not in variants:
            raise ValueError(f"unknown variant: {variant}")
        payload = deepcopy(base_config)
        payload = deep_update(payload, variants[variant])
        cfg_path = config_dir / f"{variant}.yaml"
        run_dir = output_root / variant
        stdout_log = log_dir / f"{variant}.stdout.log"
        stderr_log = log_dir / f"{variant}.stderr.log"
        dump_yaml(cfg_path, payload)

        cmd = [
            sys.executable,
            "-u",
            "analyze.py",
            "--config",
            str(cfg_path),
            "--input",
            args.input,
            "--output-dir",
            str(run_dir),
            "--plot-time-bin-width",
            str(24 * 3600),
            "--run-period",
            args.run_period_start,
            args.run_period_end,
        ]

        with stdout_log.open("w", encoding="utf-8") as out_fh, stderr_log.open("w", encoding="utf-8") as err_fh:
            proc = subprocess.run(
                cmd,
                cwd=str(WORKSPACE),
                stdout=out_fh,
                stderr=err_fh,
                text=True,
                check=False,
            )

        timestamp_dir = latest_timestamp_dir(run_dir)
        metrics = summarise_run(timestamp_dir) if timestamp_dir is not None else {}
        with summary_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    variant,
                    "ok" if proc.returncode == 0 else "failed",
                    proc.returncode,
                    str(timestamp_dir) if timestamp_dir is not None else "",
                    metrics.get("template_status", ""),
                    metrics.get("n_fitted", ""),
                    metrics.get("n_valid", ""),
                    metrics.get("n_plot_entries", ""),
                    metrics.get("chi2_mean", ""),
                    metrics.get("chi2_median", ""),
                    metrics.get("chi2_max", ""),
                    metrics.get("shift_rms_kev", ""),
                    metrics.get("shift_p95_kev", ""),
                    metrics.get("shift_max_kev", ""),
                    metrics.get("bins_with_suspicious_bound_hits", ""),
                    metrics.get("S_Po216_bound_hits", ""),
                    metrics.get("S_Bi212_bound_hits", ""),
                    metrics.get("lambda_beta_Po214_bound_hits", ""),
                    metrics.get("linear_plot_count", ""),
                    metrics.get("log_plot_count", ""),
                    metrics.get("has_bi212_series_plot", ""),
                    metrics.get("series_pngs", ""),
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
