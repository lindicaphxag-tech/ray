"""Static execution contract for Ray RLlib optimizer ownership Bazel test.

No Ray installation required. Checks BUILD registration AND Python entrypoint.
A Bazel py_test that executes the Python script without __main__ dispatch can
exit successfully without running any test_* functions.

Run from Ray checkout:
    python ci/check_optimizer_ownership_bazel_entry.py
"""
import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEST_RELPATH = "core/learner/tests/test_torch_optimizer_restore_ownership.py"
TEST = ROOT / "python/ray/rllib" / TEST_RELPATH
BUILD = ROOT / "python/ray/rllib/BUILD.bazel"


def check_source(source: str) -> None:
    tree = ast.parse(source)
    assert any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "test_torch_cpu_restored_optimizer_buffers_own_storage"
        for node in tree.body
    ), "Ownership test function missing"
    mains = [
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    ]
    assert len(mains) == 1, "Test script has no unique __main__ entry"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "pytest.main"
        for branch in mains[0].body for node in ast.walk(branch)
    ), "Main does not dispatch to pytest; Bazel could be false green"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "sys.exit"
        for branch in mains[0].body for node in ast.walk(branch)
    ), "Main does not propagate pytest failure exit code"


def check_build(source: str) -> None:
    # These are the exact Bazel target name and source path.
    pattern = (
        r'py_test\s*\(\s*name\s*=\s*'
        r'"test_torch_optimizer_restore_ownership"'
        r'[\s\S]*?srcs\s*=\s*\[\s*"'
        + re.escape(TEST_RELPATH) + r'"\s*\]'
    )
    assert re.search(pattern, source), "Ownership py_test BUILD target missing"


def main() -> None:
    source = TEST.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")
    check_source(source)
    check_build(build)
    # Negative control: the prior production test declared test_* but had
    # no __main__; it must be rejected as non-executing by this audit.
    old_source = source.split('\nif __name__ == "__main__":')[0]
    try:
        check_source(old_source)
    except AssertionError as exc:
        print("EXPECTED RED: old direct Bazel script:", exc)
    else:
        raise AssertionError("The former Bazel false-green test escaped audit")
    print("GREEN: ownership pytest is registered AND executable via Bazel")


if __name__ == "__main__":
    main()
