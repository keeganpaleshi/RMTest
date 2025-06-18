import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def test_time_bin_width_aliases():
    args_long = analyze.parse_args([
        "--config",
        "cfg.json",
        "--input",
        "data.csv",
        "--output_dir",
        "out",
        "--plot-time-bin-width",
        "5",
    ])
    args_short = analyze.parse_args([
        "--config",
        "cfg.json",
        "--input",
        "data.csv",
        "--output_dir",
        "out",
        "--time-bin-width",
        "5",
    ])
    assert args_long.time_bin_width == args_short.time_bin_width == 5.0
