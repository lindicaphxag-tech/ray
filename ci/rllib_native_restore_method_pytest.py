"""Run native RLlib unittest on current/pinned Ray source against compiled wheel.

This overlays only the production Algorithm.restore_env_runners *method body*
from a selected file onto the installed Ray 2.59.0 Algorithm class. Pytest
executes the candidate's real tests/test_algorithm.py file and initializes
Ray via its native TestAlgorithm.setUpClass. Baseline must fail on the stated
mapping-order assertion (not on importing the wrong runtime).
"""
import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import tempfile
import textwrap
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm-file", type=Path, default=Path("python/ray/rllib/algorithms/algorithm.py"))
parser.add_argument("--test-file", type=Path, default=Path("python/ray/rllib/algorithms/tests/test_algorithm.py"))
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()

import pytest
import ray
from ray.rllib.algorithms import algorithm as algorithm_module

print("native installed Ray", ray.__version__, ray._raylet.__file__, flush=True)
tree = ast.parse(args.algorithm_file.read_text(encoding="utf-8"), filename=str(args.algorithm_file))
cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Algorithm")
method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "restore_env_runners")
method.decorator_list = []
# All annotation identifiers are available in installed module; method body
# (including nested _sync_env_runner) is unchanged.
scope = vars(algorithm_module).copy()
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(args.algorithm_file),"exec"),scope)
algorithm_module.Algorithm.restore_env_runners = scope["restore_env_runners"]
print("native test method source", args.algorithm_file,flush=True)

output = io.StringIO()
# Installed PyPI Ray omits ray.tests, while the repo's parent conftest imports
# it. Copy the unchanged native test to an isolated pytest root so its test
# method executes against the released Ray wheel, not the repository conftest.
with tempfile.TemporaryDirectory(prefix="rllib-native-restore-") as root:
    test_copy = Path(root) / "test_algorithm.py"
    # Exact native unittest METHOD is retained verbatim. The surrounding
    # upstream 3.0 test module has unrelated imports absent in the 2.59 wheel.
    test_tree = ast.parse(args.test_file.read_text(encoding="utf-8"))
    original_class = next(
        n for n in test_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TestAlgorithm")
    native_method = next(
        n for n in original_class.body if isinstance(n, ast.FunctionDef)
        and n.name == "test_restore_env_runner_reinstates_runtime_multi_agent_routing")
    native_source = ast.unparse(native_method)
    test_copy.write_text(
        "import unittest\n"
        "from collections import defaultdict\n"
        "from types import SimpleNamespace\n"
        "from unittest.mock import MagicMock, patch\n"
        "from ray.rllib.algorithms.algorithm import Algorithm\n"
        "\nclass TestAlgorithm(unittest.TestCase):\n"
        + textwrap.indent(native_source,"    ") + "\n",
        encoding="utf-8")
    ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR")
    try:
        with redirect_stdout(output), redirect_stderr(output):
            code = pytest.main(["-q","-vv","--tb=short",
                "--confcutdir", root, str(test_copy),
                "-k","test_restore_env_runner_reinstates_runtime_multi_agent_routing"])
    finally:
        ray.shutdown()
text = output.getvalue()
print(text,flush=True)
if args.baseline:
    assert code == pytest.ExitCode.TESTS_FAILED, f"Expected baseline test failure, got {code}"
    assert "1 failed" in text, "Baseline did not collect the single test"
    assert "AssertionError" in text, "Baseline failure was not a test assertion"
    assert "mapping" in text or "Lists differ" in text, "Expected routing assertion to fail"
    print("EXPECTED NATIVE RED: unchanged official restore method misses runtime routing")
else:
    assert code == pytest.ExitCode.OK, f"Native regression failed with pytest code {code}"
    assert "1 passed" in text
    print("PASS NATIVE GREEN: candidate restore method meets source-weight-to-routing contract")
