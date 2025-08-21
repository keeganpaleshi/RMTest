import ast
from pathlib import Path


def _is_log(func: ast.AST) -> bool:
    return (
        isinstance(func, ast.Name) and func.id == "log"
    ) or (
        isinstance(func, ast.Attribute) and func.attr == "log"
    )


def _is_expm1(func: ast.AST) -> bool:
    return (
        isinstance(func, ast.Name) and func.id == "expm1"
    ) or (
        isinstance(func, ast.Attribute) and func.attr == "expm1"
    )


def _contains_expm1(node: ast.AST) -> bool:
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call) and _is_expm1(inner.func):
            return True
    return False


def _has_log_expm1(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_log(node.func):
            args = list(node.args) + [kw.value for kw in node.keywords]
            if any(_contains_expm1(arg) for arg in args):
                return True
    return False


def test_no_raw_log_expm1():
    repo_root = Path(__file__).resolve().parents[1]
    offending = []
    for path in repo_root.rglob("*.py"):
        if path.name == "math_utils.py":
            continue
        tree = ast.parse(path.read_text())
        if _has_log_expm1(tree):
            offending.append(str(path.relative_to(repo_root)))
    assert not offending, f"raw log(expm1()) found in: {offending}"
