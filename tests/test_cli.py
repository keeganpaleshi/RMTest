import subprocess
import sys
from pathlib import Path


def test_cli_help():
    script = Path(__file__).resolve().parents[1] / "analyze.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--background-model" in result.stdout
    assert "{linear,loglin_unit}" in result.stdout
    assert "--likelihood" in result.stdout
    assert "{current,extended}" in result.stdout
