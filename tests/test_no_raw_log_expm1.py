import pathlib
import re

ALLOWED_FILES = {
    pathlib.Path('math_utils.py'),
    pathlib.Path('tests/test_math_utils.py'),
    pathlib.Path('tests/test_no_raw_log_expm1.py'),
}

pattern = re.compile(r"log\s*\(\s*[^)]*expm1")


def test_no_raw_log_expm1_usage():
    root = pathlib.Path(__file__).resolve().parents[1]
    offending = []
    for path in root.rglob('*.py'):
        rel_path = path.relative_to(root)
        if rel_path in ALLOWED_FILES:
            continue
        text = path.read_text()
        if pattern.search(text):
            offending.append(str(rel_path))
    assert not offending, f"Raw log(expm1) usage found in: {offending}"
