import ast
import pathlib
import re
import numpy as np

from math_utils import log_expm1_stable

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {"tests", ".git", "venv", ".venv", "env"}
PATTERN = re.compile(r"(np\.log\(np\.expm1\(|log\(expm1\()")


def _iter_py_files():
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def _allowed_math_utils_lines():
    path = ROOT / "math_utils.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed = set()

    # module-level docstring
    if ast.get_docstring(tree):
        doc = tree.body[0]
        allowed.update(range(doc.lineno, doc.end_lineno + 1))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if ast.get_docstring(node):
                doc = node.body[0]
                allowed.update(range(doc.lineno, doc.end_lineno + 1))
            if node.name == "log_expm1_stable":
                allowed.update(range(node.lineno, node.end_lineno + 1))
    return allowed


ALLOWED_MATH_UTILS_LINES = _allowed_math_utils_lines()


def test_no_unsafe_expm1_usage():
    bad = []
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        for match in PATTERN.finditer(text):
            lineno = text[: match.start()].count("\n") + 1
            if (
                path.name == "math_utils.py"
                and lineno in ALLOWED_MATH_UTILS_LINES
            ):
                continue
            bad.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not bad, "Unsafe expm1 usage found:\n" + "\n".join(bad)


def test_log_expm1_stable_monotonic():
    y = np.array([-1e-8, 0.0, 1e-8, 50.0, 100.0, 500.0])
    vals = log_expm1_stable(y)
    assert np.all(np.isfinite(vals))
    diffs = np.diff(vals)
    assert np.all(diffs >= 0), "Values are not monotone increasing"
