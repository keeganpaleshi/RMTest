import ast
from pathlib import Path


def _is_log(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Name) and node.id == "log"
    ) or (
        isinstance(node, ast.Attribute) and node.attr == "log"
    )


def _contains_expm1(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Name) and func.id == "expm1"
        ) or (
            isinstance(func, ast.Attribute) and func.attr == "expm1"
        ):
            return True
        for arg in node.args:
            if _contains_expm1(arg):
                return True
        for kw in node.keywords:
            if _contains_expm1(kw.value):
                return True
        return False
    for child in ast.iter_child_nodes(node):
        if _contains_expm1(child):
            return True
    return False


def test_no_direct_log_expm1_usage():
    repo_root = Path(__file__).resolve().parents[1]
    for path in repo_root.rglob("*.py"):
        if path.name in {"math_utils.py", "test_math_utils.py", "test_no_log_expm1.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_log(node.func):
                if node.args and _contains_expm1(node.args[0]):
                    raise AssertionError(
                        f"Disallowed log(expm1(.)) usage in {path}:{node.lineno}"
                    )
