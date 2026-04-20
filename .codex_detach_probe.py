from pathlib import Path
import time

root = Path(r"C:\Users\keega\Radon\RMTest")
(root / ".codex_detach_probe_started.txt").write_text("started", encoding="utf-8")
time.sleep(15)
(root / ".codex_detach_probe_done.txt").write_text("done", encoding="utf-8")
