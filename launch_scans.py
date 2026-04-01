"""Launch I-series scans as detached processes that survive parent death."""
import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

scans = [
    ("I14m9c", "scan_configs/I14m9_config.yaml"),  # passive Po216, no DNL, quad asym + tail
]

for scan_id, config in scans:
    log_path = os.path.join(BASE, "results", f"{scan_id}_analyze.log")
    output_dir = os.path.join(BASE, "results", f"scan_{scan_id}")
    cmd = [
        PYTHON, os.path.join(BASE, "analyze.py"),
        "--config", os.path.join(BASE, config),
        "--output-dir", output_dir,
    ]
    with open(log_path, "w") as log_file:
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=BASE,
            creationflags=flags,
        )
        print(f"Launched {scan_id}: PID={p.pid}, log={log_path}")
