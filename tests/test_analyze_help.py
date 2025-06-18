import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def test_analyze_help_exits(capsys):
    with pytest.raises(SystemExit) as excinfo:
        analyze.main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()
