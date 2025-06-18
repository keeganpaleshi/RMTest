import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def test_time_bin_width_alias():
    base = ["--config", "c", "--input", "i", "--output_dir", "o"]
    args1 = analyze.parse_args(base + ["--plot-time-bin-width", "5"])
    args2 = analyze.parse_args(base + ["--time-bin-width", "5"])
    assert args1.time_bin_width == 5.0
    assert args2.time_bin_width == 5.0


