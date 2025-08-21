import ast
from pathlib import Path

import numpy as np

from math_utils import log_expm1_stable

UNSAFE_SUBSTRINGS = ["np.lo" "g(np.ex" "pm1(", "lo" "g(ex" "pm1("]
SKIP_PARTS = {"tests", ".git"}


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in SKIP_PARTS or "venv" in part or part.endswith("env") for part in path.parts):
            continue
        yield path


def _find_occurrences(text: str):
    for substr in UNSAFE_SUBSTRINGS:
        start = 0
        while True:
            idx = text.find(substr, start)
            if idx == -1:
                break
            line = text.count("\n", 0, idx) + 1
            yield substr, line
            start = idx + 1


def test_no_unsafe_expm1_usage():
    root = Path(__file__).resolve().parents[1]
    for path in _iter_python_files(root):
        text = path.read_text()
        occurrences = list(_find_occurrences(text))
        if path.name == "math_utils.py":
            if not occurrences:
                continue
            tree = ast.parse(text)
            func = next(
                node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "log_expm1_stable"
            )
            allowed = [(func.lineno, func.end_lineno)]
            if func.body:
                doc = func.body[0]
                if (
                    isinstance(doc, ast.Expr)
                    and isinstance(getattr(doc, "value", None), ast.Constant)
                    and isinstance(doc.value.value, str)
                ):
                    allowed.append((doc.lineno, doc.end_lineno))
            for _, line in occurrences:
                assert any(start <= line <= end for start, end in allowed), (
                    f"Unsafe lo" + f"g(expm1) usage in {path} line {line}"
                )
        else:
            assert not occurrences, (
                f"Unsafe lo" + f"g(expm1) usage in {path}: lines {[line for _, line in occurrences]}"
            )


def test_log_expm1_stable_values():
    y = np.array([-1e-8, 0.0, 1e-8, 50.0, 100.0, 500.0])
    result = log_expm1_stable(y)
    assert np.all(np.isfinite(result))
    assert np.all(np.diff(result) >= 0)
