from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(r"C:\Users\keega\Radon\RMTest")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def default_windows() -> list[tuple[str, str, str]]:
    return [
        ("6mo", "2024-06-01T00:00:00Z", "2024-12-01T00:00:00Z"),
        ("3mo", "2024-09-01T00:00:00Z", "2024-12-01T00:00:00Z"),
        ("1mo", "2024-11-01T00:00:00Z", "2024-12-01T00:00:00Z"),
        ("2wk", "2024-11-17T00:00:00Z", "2024-12-01T00:00:00Z"),
    ]


def write_status(
    path: Path,
    *,
    phase: str,
    windows: list[tuple[str, str, str]],
    hours: str,
    base_config: str,
    input_csv: str,
    current_label: str | None = None,
    current_run_dir: str = "",
    current_summary_csv: str = "",
) -> None:
    payload = {
        "phase": phase,
        "windows": [
            {"label": label, "start": start, "end": end}
            for label, start, end in windows
        ],
        "hours": hours,
        "base_config": base_config,
        "input_csv": input_csv,
        "current_label": current_label,
        "current_run_dir": current_run_dir,
        "current_summary_csv": current_summary_csv,
        "updated_at": now_utc().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=str(WORKSPACE / ".codex_run_merged_output_template_review_v3.yaml"))
    parser.add_argument("--input", default=str(WORKSPACE / "merged_output.csv"))
    parser.add_argument("--hours", default="48,36,24,18,14,12,10,8,6,4,2")
    parser.add_argument("--output-prefix", default=".codex_run_template_multiwindow")
    args = parser.parse_args()

    session_id = now_utc().strftime("%Y%m%dT%H%M%SZ")
    root = WORKSPACE / f"{args.output_prefix}_{session_id}"
    root.mkdir(parents=True, exist_ok=True)
    manifest_csv = root / "window_manifest.csv"
    status_json = root / "status.json"

    windows = default_windows()
    with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "label",
                "start",
                "end",
                "status",
                "exit_code",
                "run_dir",
                "summary_csv",
                "started_at_utc",
                "ended_at_utc",
            ]
        )

    write_status(
        status_json,
        phase="starting",
        windows=windows,
        hours=args.hours,
        base_config=args.base_config,
        input_csv=args.input,
    )

    for label, start, end in windows:
        sweep_prefix = f"{args.output_prefix}_{session_id}_{label}"
        started_at = now_utc().isoformat()
        expected_root_prefix = WORKSPACE / sweep_prefix

        write_status(
            status_json,
            phase="running",
            windows=windows,
            hours=args.hours,
            base_config=args.base_config,
            input_csv=args.input,
            current_label=label,
            current_run_dir=str(expected_root_prefix),
            current_summary_csv=str(expected_root_prefix) + "_*/sweep_summary.csv",
        )

        cmd = [
            sys.executable,
            ".codex_run_template_review_sweep.py",
            "--base-config",
            args.base_config,
            "--input",
            args.input,
            "--hours",
            args.hours,
            "--output-prefix",
            sweep_prefix,
            "--run-period-start",
            start,
            "--run-period-end",
            end,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(WORKSPACE),
            check=False,
        )

        run_dirs = sorted(WORKSPACE.glob(f"{sweep_prefix}_*"))
        run_dir = str(run_dirs[-1]) if run_dirs else ""
        summary_csv = str(Path(run_dir) / "sweep_summary.csv") if run_dir else ""
        ended_at = now_utc().isoformat()
        with manifest_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    label,
                    start,
                    end,
                    "ok" if proc.returncode == 0 else "failed",
                    proc.returncode,
                    run_dir,
                    summary_csv,
                    started_at,
                    ended_at,
                ]
            )

    write_status(
        status_json,
        phase="completed",
        windows=windows,
        hours=args.hours,
        base_config=args.base_config,
        input_csv=args.input,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
