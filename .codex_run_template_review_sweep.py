from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(r"C:\Users\keega\Radon\RMTest")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_hours(raw: str) -> list[int]:
    hours: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        hours.append(int(part))
    if not hours:
        raise ValueError("at least one hour value is required")
    return hours


def latest_timestamp_dir(run_dir: Path) -> Path | None:
    candidates = sorted([p for p in run_dir.iterdir() if p.is_dir()]) if run_dir.exists() else []
    return candidates[-1] if candidates else None


def safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def count_template_plots(template_plot_dir: Path | None) -> tuple[int, int]:
    if template_plot_dir is None or not template_plot_dir.exists():
        return 0, 0
    base_pngs = [
        p for p in template_plot_dir.glob("template_fit_bin_*.png")
        if not p.name.endswith("_log.png")
    ]
    log_pngs = list(template_plot_dir.glob("*_log.png"))
    return len(base_pngs), len(log_pngs)


def write_status(
    *,
    path: Path,
    session_id: str,
    phase: str,
    hours: list[int],
    base_config: Path,
    input_csv: Path,
    summary_csv: Path,
    run_period_start: str | None,
    run_period_end: str | None,
    current_hour: int | None = None,
    current_run_dir: str = "",
    current_stdout_log: str = "",
    current_stderr_log: str = "",
) -> None:
    payload = {
        "session_id": session_id,
        "requested_hours": hours,
        "phase": phase,
        "current_hour": current_hour,
        "current_run_dir": current_run_dir,
        "current_stdout_log": current_stdout_log,
        "current_stderr_log": current_stderr_log,
        "base_config": str(base_config),
        "input_csv": str(input_csv),
        "summary_csv": str(summary_csv),
        "run_period_start": run_period_start,
        "run_period_end": run_period_end,
        "updated_at": now_utc().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hours", default="24,12,8,4")
    parser.add_argument("--output-prefix", default=".codex_run_template_review_sweep")
    parser.add_argument("--run-period-start")
    parser.add_argument("--run-period-end")
    args = parser.parse_args()

    base_config = Path(args.base_config)
    input_csv = Path(args.input)
    hours = parse_hours(args.hours)
    session_id = now_utc().strftime("%Y%m%dT%H%M%SZ")
    sweep_root = WORKSPACE / f"{args.output_prefix}_{session_id}"
    config_dir = sweep_root / "configs"
    log_dir = sweep_root / "logs"
    output_root = sweep_root / "outputs"
    summary_csv = sweep_root / "sweep_summary.csv"
    status_json = sweep_root / "status.json"

    for path in (sweep_root, config_dir, log_dir, output_root):
        path.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "hour",
                "bin_seconds",
                "status",
                "exit_code",
                "run_dir",
                "timestamp_dir",
                "template_plot_dir",
                "base_plot_count",
                "log_plot_count",
                "template_status",
                "n_fitted",
                "n_valid",
                "n_plot_entries",
                "bins_with_any_bound_hits",
                "bins_with_suspicious_bound_hits",
                "start_utc",
                "end_utc",
            ]
        )

    write_status(
        path=status_json,
        session_id=session_id,
        phase="starting",
        hours=hours,
        base_config=base_config,
        input_csv=input_csv,
        summary_csv=summary_csv,
        run_period_start=args.run_period_start,
        run_period_end=args.run_period_end,
    )

    for hour in hours:
        bin_seconds = int(hour) * 3600
        hour_tag = f"{int(hour):02d}h"
        run_dir = output_root / f"template_bin_{hour_tag}"
        stdout_log = log_dir / f"template_bin_{hour_tag}.stdout.log"
        stderr_log = log_dir / f"template_bin_{hour_tag}.stderr.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-u",
            "analyze.py",
            "--config",
            str(base_config),
            "--input",
            str(input_csv),
            "--output-dir",
            str(run_dir),
            "--plot-time-bin-width",
            str(bin_seconds),
        ]
        if args.run_period_start and args.run_period_end:
            cmd.extend(["--run-period", args.run_period_start, args.run_period_end])

        write_status(
            path=status_json,
            session_id=session_id,
            phase="running",
            hours=hours,
            base_config=base_config,
            input_csv=input_csv,
            summary_csv=summary_csv,
            run_period_start=args.run_period_start,
            run_period_end=args.run_period_end,
            current_hour=int(hour),
            current_run_dir=str(run_dir),
            current_stdout_log=str(stdout_log),
            current_stderr_log=str(stderr_log),
        )

        start_utc = now_utc().isoformat()
        with stdout_log.open("w", encoding="utf-8") as out_fh, stderr_log.open("w", encoding="utf-8") as err_fh:
            proc = subprocess.run(
                cmd,
                cwd=str(WORKSPACE),
                stdout=out_fh,
                stderr=err_fh,
                text=True,
                check=False,
            )
        end_utc = now_utc().isoformat()

        timestamp_dir = latest_timestamp_dir(run_dir)
        template_plot_dir = timestamp_dir / "template_bin_fits" if timestamp_dir is not None else None
        base_plot_count, log_plot_count = count_template_plots(template_plot_dir)

        summary_obj = safe_json(timestamp_dir / "summary.json") if timestamp_dir is not None else {}
        template_summary = summary_obj.get("template_fitting", {}) if isinstance(summary_obj, dict) else {}

        with summary_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    int(hour),
                    bin_seconds,
                    "ok" if proc.returncode == 0 else "failed",
                    proc.returncode,
                    str(run_dir),
                    str(timestamp_dir) if timestamp_dir is not None else "",
                    str(template_plot_dir) if template_plot_dir is not None and template_plot_dir.exists() else "",
                    base_plot_count,
                    log_plot_count,
                    template_summary.get("status", ""),
                    template_summary.get("n_fitted", ""),
                    template_summary.get("n_valid", ""),
                    template_summary.get("n_plot_entries", ""),
                    template_summary.get("bins_with_any_bound_hits", ""),
                    template_summary.get("bins_with_suspicious_bound_hits", ""),
                    start_utc,
                    end_utc,
                ]
            )

    write_status(
        path=status_json,
        session_id=session_id,
        phase="completed",
        hours=hours,
        base_config=base_config,
        input_csv=input_csv,
        summary_csv=summary_csv,
        run_period_start=args.run_period_start,
        run_period_end=args.run_period_end,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
