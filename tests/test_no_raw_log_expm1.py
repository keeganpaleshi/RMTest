import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {REPO_ROOT / "math_utils.py"}


def _find_raw_log_expm1():
    for path in REPO_ROOT.rglob("*.py"):
        if path in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                else:
                    continue
                if name != "log" or not node.args:
                    continue
                arg = node.args[0]
                if isinstance(arg, ast.Call):
                    inner = arg.func
                    if isinstance(inner, ast.Attribute):
                        inner_name = inner.attr
                    elif isinstance(inner, ast.Name):
                        inner_name = inner.id
                    else:
                        continue
                    if inner_name == "expm1":
                        yield path.relative_to(REPO_ROOT), node.lineno


def test_no_raw_log_expm1():
    offenders = list(_find_raw_log_expm1())
    assert not offenders, f"Found raw log(expm1()) usage in: {offenders}"
