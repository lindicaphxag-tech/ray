"""Audit Learner.set_state's version-zero checkpoint round trip, using source AST.

Unlike a hand-written version-counter imitation, this executes the actual
unmodified Learner.set_state method body from the current fork checkout.
A tiny fake learner substitutes only dependencies, not the method under test.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import subprocess
import unittest


PATH = "python/ray/rllib/core/learner/learner.py"
BASE = "95d3a43269d92ed2f255801399abc428d580600b"
WEIGHTS_SEQ_NO = "weights_seq_no"
COMPONENT_RL_MODULE = "rl_module"
COMPONENT_OPTIMIZER = "optimizer"
COMPONENT_METRICS_LOGGER = "metrics_logger"


class Module:
    def __init__(self):
        self.state = {"main": {"weight": 0}}
        self.calls = 0

    def set_state(self, state):
        self.state = state
        self.calls += 1


class FakeLearner:
    def __init__(self, seq=0):
        self._weights_seq_no = seq
        self.module = Module()
        self.config = SimpleNamespace(multi_agent=lambda **_: None)
        self.metrics = SimpleNamespace(set_state=lambda _: None)

    def _check_is_built(self):
        return True

    def _set_optimizer_state(self, state):
        pass


def method_from_source(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Learner")
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "set_state")
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    mod = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    ns = {
        "WEIGHTS_SEQ_NO": WEIGHTS_SEQ_NO,
        "COMPONENT_RL_MODULE": COMPONENT_RL_MODULE,
        "COMPONENT_OPTIMIZER": COMPONENT_OPTIMIZER,
        "COMPONENT_METRICS_LOGGER": COMPONENT_METRICS_LOGGER,
    }
    exec(compile(mod, "<actual Learner.set_state AST>", "exec"), ns)
    return ns["set_state"]


def examine(fn):
    def invoke(initial, state):
        learner = FakeLearner(initial)
        fn(learner, state)
        return {
            "seq": learner._weights_seq_no,
            "module_calls": learner.module.calls,
            "weight": learner.module.state["main"]["weight"],
        }

    checkpoint_zero = {
        WEIGHTS_SEQ_NO: 0, COMPONENT_RL_MODULE: {"main": {"weight": 4}}
    }
    manual_update = {COMPONENT_RL_MODULE: {"main": {"weight": 5}}}
    stale_version = {
        WEIGHTS_SEQ_NO: 4, COMPONENT_RL_MODULE: {"main": {"weight": 7}}
    }
    return {
        "explicit_zero_checkpoint": invoke(0, checkpoint_zero),
        "unversioned_manual_mutation": invoke(0, manual_update),
        "stale_versioned_state": invoke(5, stale_version),
        "explicit_zero_on_later_version": invoke(5, checkpoint_zero),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--require-fixed", action="store_true")
    args = parser.parse_args()
    source = Path(PATH).read_text()
    now = examine(method_from_source(source))
    expected = {
        "explicit_zero_checkpoint": {"seq": 0, "module_calls": 1, "weight": 4},
        "unversioned_manual_mutation": {"seq": 1, "module_calls": 1, "weight": 5},
        "stale_versioned_state": {"seq": 5, "module_calls": 0, "weight": 0},
        "explicit_zero_on_later_version": {"seq": 6, "module_calls": 1, "weight": 4},
    }
    if args.compare:
        base = subprocess.run(["git", "show", f"{BASE}:{PATH}"],
                              capture_output=True, text=True, check=True).stdout
        original = examine(method_from_source(base))
        print("BEFORE_PR", json.dumps(original, sort_keys=True), flush=True)
    print("CURRENT_PR", json.dumps(now, sort_keys=True), flush=True)
    mismatches = {k: {"got": now[k], "want": v}
                  for k, v in expected.items() if now[k] != v}
    print("REGRESSION_MISMATCHES", json.dumps(mismatches, sort_keys=True))
    if args.require_fixed:
        if mismatches:
            raise SystemExit("regression present")
        print("SOURCE_METHOD_GREEN")
    else:
        if "explicit_zero_checkpoint" not in mismatches:
            raise SystemExit("expected zero-checkpoint regression was not reproduced")
        print("ZERO_CHECKPOINT_REGRESSION_REPRODUCED")


if __name__ == "__main__":
    main()
