import pathlib
import re

import numpy as np
from math_utils import log_expm1_stable

FORBIDDEN = ("np.log(np.expm1(", "log(expm1(")
SKIP_DIRS = {"tests", ".git", "venv", ".venv", "env", ".env"}



def _allowed_lines_math_utils(path: pathlib.Path) -> set[int]:
    import ast

    source = path.read_text()
    tree = ast.parse(source)
    allowed: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "log_expm1_stable":
            # include the whole function body
            allowed.update(range(node.lineno, node.end_lineno + 1))
            # include docstring if present
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                node.body[0].value, ast.Constant
            ) and isinstance(node.body[0].value.value, str):
                allowed.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
            break
    return allowed


def test_no_unsafe_expm1_usage():
    root = pathlib.Path(__file__).resolve().parents[1]
    offending: list[str] = []
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & SKIP_DIRS:
            continue
        text = path.read_text()
        if path.name == "math_utils.py":
            allowed = _allowed_lines_math_utils(path)
            for sub in FORBIDDEN:
                for m in re.finditer(re.escape(sub), text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    if line_no not in allowed:
                        offending.append(f"{path}:{line_no}")
        else:
            if any(sub in text for sub in FORBIDDEN):
                offending.append(str(path))
    assert not offending, f"Unsafe log(expm1) usage found in: {offending}"


def test_log_expm1_stable_sanity():
    y = np.array([-1e-8, 0.0, 1e-8, 50.0, 100.0, 500.0])
    vals = log_expm1_stable(y)
    assert np.all(np.isfinite(vals)), "Values must be finite"
    diffs = np.diff(vals)
    assert np.all(diffs >= 0), "Values must be non-decreasing"
