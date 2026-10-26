"""Source-exact, dependency-free regression for Ray SingleAgentEnvRunner versioning.

Run from a Ray checkout: python ci/rllib_single_agent_weight_version_contract.py
Compiles the real SingleAgentEnvRunner.set_state AST without Ray's compiled
extensions, exercising old-code RED / current-code GREEN. Native RLlib tests
remain separately required for upstream acceptance.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from unittest.mock import MagicMock

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "python/ray/rllib/env/single_agent_env_runner.py"
)
code = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
runner = next(
    x for x in code.body
    if isinstance(x, ast.ClassDef) and x.name == "SingleAgentEnvRunner"
)
method = next(
    x for x in runner.body
    if isinstance(x, ast.FunctionDef) and x.name == "set_state"
)


def build_method(*, revert_monotonic: bool):
    node = copy.deepcopy(method)
    node.decorator_list = []
    node.returns = None
    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    targets = [
        x for x in ast.walk(node)
        if isinstance(x, ast.If)
        and ast.unparse(x.test) == "weights_seq_no > self._weights_seq_no"
    ]
    assert len(targets) == 1, "Expected exactly one monotonic version gate"
    if revert_monotonic:
        targets[0].test = ast.parse("weights_seq_no > 0", mode="eval").body
    executable = ast.fix_missing_locations(
        ast.Module(body=[node], type_ignores=[])
    )
    ns = {
        "COMPONENT_ENV_TO_MODULE_CONNECTOR": "env_to_module",
        "COMPONENT_MODULE_TO_ENV_CONNECTOR": "module_to_env",
        "COMPONENT_RL_MODULE": "module",
        "DEFAULT_MODULE_ID": "default",
        "NUM_ENV_STEPS_SAMPLED_LIFETIME": "lifetime",
        "WEIGHTS_SEQ_NO": "version",
        "ray": type(
            "Ray", (),
            {"ObjectRef": type("ObjectRef", (), {}),
             "get": staticmethod(lambda x: x)},
        ),
    }
    exec(compile(executable, filename=str(SOURCE), mode="exec"), ns)
    return ns["set_state"]


def check(fn):
    instance = MagicMock()
    instance._weights_seq_no = 8
    for number in (7, 8):
        fn(instance, {
            "module": {"default": {"weight": number}},
            "version": number,
        })
    instance.module.set_state.assert_not_called()
    assert instance._weights_seq_no == 8, (
        "Rejected weight state downgraded sequence",
        instance._weights_seq_no,
    )
    fn(instance, {
        "module": {"default": {"weight": 9}}, "version": 9,
    })
    instance.module.set_state.assert_called_once_with({"weight": 9})
    assert instance._weights_seq_no == 9
    instance.module.set_state.reset_mock()
    fn(instance, {"module": {"default": {"weight": 100}}})
    instance.module.set_state.assert_called_once_with({"weight": 100})
    assert instance._weights_seq_no == 9


try:
    check(build_method(revert_monotonic=True))
except AssertionError as exc:
    print("EXPECTED RED: old source gate fails:", exc)
else:
    raise AssertionError("Old implementation unexpectedly passed")
check(build_method(revert_monotonic=False))
print("GREEN: source-extracted current gate passes stale/equal/new/force")
