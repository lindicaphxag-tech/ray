"""Execution-level witness that Bazel's plain py_test entry invokes pytest.

Takes the repository ownership test file's AST, replaces its test body with
a sentinel failure, and invokes the resulting file directly with Python.
A file without __main__ exits 0 without reaching the test. With pytest.main
and sys.exit, it must exit nonzero *because the sentinel test is executed*.
This tests collection/exit semantics, NOT RLlib's checkpoint correctness.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import subprocess
import sys
import tempfile


def check(path: Path) -> None:
    original = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((
        copy.deepcopy(x)
        for x in original.body
        if isinstance(x, ast.FunctionDef)
        and x.name == "test_torch_cpu_restored_optimizer_buffers_own_storage"
    ), None)
    assert fn is not None, "Ownership test definition missing"
    fn.decorator_list = []
    fn.body = [
        ast.Raise(exc=ast.Call(
            func=ast.Name(id="AssertionError", ctx=ast.Load()),
            args=[ast.Constant(value="SENTINEL_TEST_WAS_EXECUTED")],
            keywords=[],
        ))
    ]
    main = next((
        copy.deepcopy(x)
        for x in original.body
        if isinstance(x, ast.If)
        and ast.unparse(x.test) == "__name__ == '__main__'"
    ), None)
    assert main is not None, "Ownership test missing pytest main"
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "test_ownership_direct_entry.py"
        for label, program in (("historical", [fn]), ("fixed", [fn, main])):
            target.write_text(
                ast.unparse(ast.fix_missing_locations(
                    ast.Module(body=program, type_ignores=[])
                )) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(target)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            combined = result.stdout + result.stderr
            observed = "SENTINEL_TEST_WAS_EXECUTED" in combined
            if label == "historical":
                assert result.returncode == 0 and not observed, (
                    label, result.returncode, combined[-2000:]
                )
            else:
                assert result.returncode == 1 and observed, (
                    label, result.returncode, combined[-2000:]
                )
            print(label, "exit", result.returncode, "sentinel executed", observed)
    print("PASS: new direct entry dispatches pytest and propagates failure")


if __name__ == "__main__":
    check(Path(
        "python/ray/rllib/core/learner/tests/"
        "test_torch_optimizer_restore_ownership.py"
    ))
