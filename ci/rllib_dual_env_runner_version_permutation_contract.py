"""Deterministic cross-runner version-ordering contract, no Ray import required.

In an editable Ray checkout (with BOTH single/multi patches), run:
  python ci/rllib_dual_env_runner_version_permutation_contract.py

Extracts the actual set_state methods from the checked-out source; no mocked
implementation substitutes for the production method body. The IO calls to
RLModule and connectors are replaced with mocks for a fast contract witness.
This is not equivalent to a distributed RLlib native integration test.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "single": ROOT / "python/ray/rllib/env/single_agent_env_runner.py",
    "multi": ROOT / "python/ray/rllib/env/multi_agent_env_runner.py",
}
CLASSES = {
    "single": "SingleAgentEnvRunner",
    "multi": "MultiAgentEnvRunner",
}

def extract(kind, *, historical=False, method="set_state"):
    path = SOURCES[kind]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    klass = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                 and x.name == CLASSES[kind])
    fn = copy.deepcopy(next(x for x in klass.body
                            if isinstance(x, ast.FunctionDef)
                            and x.name == method))
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    if method == "set_state":
        guards = [
            x for x in ast.walk(fn)
            if isinstance(x, ast.If)
            and ast.unparse(x.test) == "weights_seq_no > self._weights_seq_no"
        ]
        assert len(guards) == 1, (kind, len(guards))
        if historical:
            guards[0].test = ast.parse("weights_seq_no > 0", mode="eval").body
    else:
        assert method == "get_state" and not historical
    namespace = {
        "COMPONENT_ENV_TO_MODULE_CONNECTOR": "input_connector",
        "COMPONENT_MODULE_TO_ENV_CONNECTOR": "output_connector",
        "COMPONENT_RL_MODULE": "module",
        "DEFAULT_MODULE_ID": "default",
        "NUM_ENV_STEPS_SAMPLED_LIFETIME": "lifetime",
        "WEIGHTS_SEQ_NO": "version",
        "ray": SimpleNamespace(
            ObjectRef=type("DummyObjectRef", (), {}),
            get=lambda ref: ref,
        ),
    }
    code = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(code, filename=str(path), mode="exec"), namespace)
    return namespace["set_state"]

def trace(fn, export, kind, sequence):
    instance = MagicMock()
    instance._weights_seq_no = 8
    accepted = 8
    for pos, version in enumerate(sequence):
        payload = {"default": {"weight": 100 + pos}}
        state = {"module": payload}
        if version is not None:
            state["version"] = version
        calls_before = instance.module.set_state.call_count
        should_install = version is None or version == 0 or version > accepted
        fn(instance, state)
        calls_after = instance.module.set_state.call_count
        assert calls_after - calls_before == int(should_install), (
            kind, pos, "unexpected module update", version,
            accepted, calls_before, calls_after,
        )
        if should_install:
            expected_arg = payload["default"] if kind == "single" else payload
            assert instance.module.set_state.call_args.args[0] == expected_arg
        if version is not None and version > accepted:
            accepted = version
        assert instance._weights_seq_no == accepted, (
            kind, pos, "positive version label regressed",
            version, accepted, instance._weights_seq_no,
        )
        assert export(instance)["version"] == accepted, (
            kind, pos, "exported version diverges from accepted sequence",
            version, accepted, export(instance)["version"],
        )
    return len(sequence)

def main():
    rng = random.Random(20261027)
    sequences = [
        [7, 8, 9, 0, None, 8, 11, 10],
        [9, 7, 9, 8, 11, 11, None],
    ]
    # Guarantee coverage within *each* permutation; a random draw alone
    # can accidentally omit a required case.
    mandatory = [None, 0, 7, 8, 9, 11, 12]
    for _ in range(80):
        seq = mandatory + [
            rng.choice([None, 0, 1, 5, 7, 8, 9, 10, 11, 12])
            for _ in range(24 - len(mandatory))
        ]
        rng.shuffle(seq)
        sequences.append(seq)
    for kind in ("single", "multi"):
        fixed = extract(kind)
        historical = extract(kind, historical=True)
        export = extract(kind, method="get_state")
        try:
            trace(historical, export, kind, [7, 8])
        except AssertionError:
            print("EXPECTED RED", kind, "rejected stale version downgrades")
        else:
            raise AssertionError("Old-version bug not reproduced: " + kind)
        transitions = sum(trace(fixed, export, kind, seq) for seq in sequences)
        print("GREEN", kind, len(sequences), "sequences,", transitions,
              "state transitions")
    print("Scope: source-extracted methods; no native Ray actor/optimizer pass")
    # A positive counter alone does not certify identity after an unversioned
    # forced install. This intentionally documents, rather than 'fixes',
    # a distinct existing compatibility contract.
    for kind in ("single", "multi"):
        fn = extract(kind)
        resident = MagicMock()
        resident._weights_seq_no = 8
        payload_a = {"default": {"weight": 8}}
        payload_b = {"default": {"weight": 123}}
        fn(resident, {"module": payload_a})
        before = resident.module.set_state.call_args.args[0]
        fn(resident, {"module": payload_b})
        after = resident.module.set_state.call_args.args[0]
        assert before != after
        assert resident._weights_seq_no == 8
        print("EXPECTED LIMITATION", kind,
              "unversioned force changes weights without updating label")

if __name__ == "__main__":
    main()
