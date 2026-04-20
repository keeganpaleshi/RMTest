from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(r"C:\Users\keega\Radon\RMTest")
BASE_CONFIG = WORKSPACE / ".codex_run_merged_output_config.yaml"
INPUT_CSV = WORKSPACE / "merged_output.csv"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def write_status(
    *,
    path: Path,
    session_id: str,
    phase: str,
    current_hour: int,
    current_run_dir: str,
    summary_csv: Path,
) -> None:
    payload = {
        "session_id": session_id,
        "phase": phase,
        "current_hour": current_hour,
        "current_run_dir": current_run_dir,
        "base_config": str(BASE_CONFIG),
        "input_csv": str(INPUT_CSV),
        "summary_csv": str(summary_csv),
        "updated_at": now_utc().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def replace_bin_width(lines: list[str], bin_seconds: int) -> list[str]:
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("plot_time_bin_width_s:"):
            out.append(f"  plot_time_bin_width_s: {bin_seconds}")
            replaced = True
        else:
            out.append(line.rstrip("\n"))
    if not replaced:
        raise RuntimeError("Could not find plot_time_bin_width_s in base config")
    return out


def main() -> int:
    session_id = now_utc().strftime("%Y%m%dT%H%M%SZ")
    sweep_root = WORKSPACE / f".codex_run_template_bin_sweep_{session_id}"
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
                "start_utc",
                "end_utc",
            ]
        )

    base_lines = BASE_CONFIG.read_text(encoding="utf-8").splitlines()
    write_status(
        path=status_json,
        session_id=session_id,
        phase="starting",
        current_hour=0,
        current_run_dir="",
        summary_csv=summary_csv,
    )

    for hour in range(1, 25):
        bin_seconds = hour * 3600
        hour_tag = f"{hour:02d}h"
        config_path = config_dir / f"template_bin_{hour_tag}.yaml"
        run_dir = output_root / f"template_bin_{hour_tag}"
        stdout_log = log_dir / f"template_bin_{hour_tag}.stdout.log"
        stderr_log = log_dir / f"template_bin_{hour_tag}.stderr.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path.write_text(
            "\n".join(replace_bin_width(base_lines, bin_seconds)) + "\n",
            encoding="utf-8",
        )

        write_status(
            path=status_json,
            session_id=session_id,
            phase="running",
            current_hour=hour,
            current_run_dir=str(run_dir),
            summary_csv=summary_csv,
        )

        start_utc = now_utc().isoformat()
        with stdout_log.open("w", encoding="utf-8") as out_fh, stderr_log.open(
            "w", encoding="utf-8"
        ) as err_fh:
            proc = subprocess.run(
                [
                    sys.executable,
                    "analyze.py",
                    "--config",
                    str(config_path),
                    "--input",
                    str(INPUT_CSV),
                    "--output-dir",
                    str(run_dir),
                ],
                cwd=str(WORKSPACE),
                stdout=out_fh,
                stderr=err_fh,
                text=True,
                check=False,
            )
        end_utc = now_utc().isoformat()

        timestamp_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
        timestamp_dir = timestamp_dirs[-1] if timestamp_dirs else None
        template_plot_dir = (
            timestamp_dir / "template_bin_fits"
            if timestamp_dir is not None
            else None
        )

        base_plot_count = 0
        log_plot_count = 0
        if template_plot_dir is not None and template_plot_dir.exists():
            base_plot_count = len(
                [p for p in template_plot_dir.glob("template_fit_bin_*.png") if not p.name.endswith("_log.png")]
            )
            log_plot_count = len(list(template_plot_dir.glob("*_log.png")))

        with summary_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    hour,
                    bin_seconds,
                    "ok" if proc.returncode == 0 else "failed",
                    proc.returncode,
                    str(run_dir),
                    str(timestamp_dir) if timestamp_dir is not None else "",
                    str(template_plot_dir) if template_plot_dir is not None and template_plot_dir.exists() else "",
                    base_plot_count,
                    log_plot_count,
                    start_utc,
                    end_utc,
                ]
            )

    write_status(
        path=status_json,
        session_id=session_id,
        phase="completed",
        current_hour=24,
        current_run_dir="",
        summary_csv=summary_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
