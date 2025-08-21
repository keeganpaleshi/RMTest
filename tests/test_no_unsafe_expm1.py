import numpy as np
import pathlib

from math_utils import log_expm1_stable


UNSAFE_PATTERNS = ["np.log(np.expm1(", "log(expm1("]


def python_files(root: pathlib.Path):
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        parts = rel.parts
        if parts[0] == "tests" or ".git" in parts or any("venv" in p for p in parts):
            continue
        yield path, rel


def test_no_unsafe_expm1_usage():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    bad = []
    for path, rel in python_files(repo_root):
        if rel.name == "math_utils.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in UNSAFE_PATTERNS:
            if pattern in text:
                bad.append(f"{rel}: {pattern}")
    assert not bad, "Unsafe log(expm1) usage found:\n" + "\n".join(bad)


def test_log_expm1_stable_sanity():
    ys = np.array([-1e-8, 0.0, 1e-8, 50.0, 100.0, 500.0])
    values = log_expm1_stable(ys)
    assert np.all(np.isfinite(values))
    diffs = np.diff(values)
    assert np.all(diffs >= 0), "log_expm1_stable should be monotone increasing"
