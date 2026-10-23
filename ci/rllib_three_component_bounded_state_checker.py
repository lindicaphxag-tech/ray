"""Bounded executable check of RLlib's THREE coupled learner state components.

Tests the actual Learner.set_state source body against an adversarial packet
space, not an imitation. Verifies the causal admission invariants for
parameters, optimizer state and trainable-policy selectors under all ordered
traces of length 1..3 over a finite 52-packet model.

No claim of general distributed linearizability, exception-safe rollback,
real GRPO returns or arbitrary-length model checking.
"""
from __future__ import annotations

import ast
import copy
import itertools
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

source = Path(sys.argv[1])
baseline = "--baseline" in sys.argv
tree = ast.parse(source.read_text(), filename=str(source))
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Learner")
fn = copy.deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "set_state"))
fn.decorator_list = []
fn.returns = None
fn.args.args[1].annotation = None
scope = {
    "COMPONENT_RL_MODULE": "module",
    "COMPONENT_OPTIMIZER": "optimizer",
    "COMPONENT_METRICS_LOGGER": "metrics",
    "WEIGHTS_SEQ_NO": "version",
}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
             str(source), "exec"), scope)
set_state = scope["set_state"]

class Model:
    def __init__(self, learner):
        self.learner = learner
    def set_state(self, state):
        self.learner.weights = state["id"]

class Config:
    def __init__(self, learner):
        self.learner = learner
    def multi_agent(self, *, policies_to_train):
        self.learner.selector = policies_to_train

class Learner:
    def __init__(self):
        self._weights_seq_no = 1
        self.weights = "local:1"
        self.optimizer = "local:1"
        self.selector = "local:1"
        self.module = Model(self)
        self.config = Config(self)
        self.metrics = SimpleNamespace(set_state=lambda *_: None)
    def _check_is_built(self):
        pass
    def _set_optimizer_state(self, state):
        self.optimizer = state["id"]
    def snapshot(self):
        return (self._weights_seq_no, self.weights, self.optimizer, self.selector)

def packets():
    kinds = ("full", "pair", "module", "module_selector", "optimizer", "selector")
    res = []
    for kind in kinds:
        for version in (0, 1, 2, 3):
            for origin in ("A", "B"):
                token = f"{origin}:{version}"
                packet = {"version": version}
                if kind in ("full", "pair", "module", "module_selector"):
                    packet["module"] = {"id": token}
                if kind in ("full", "pair", "optimizer"):
                    packet["optimizer"] = {"id": token}
                if kind in ("full", "module_selector", "selector"):
                    packet["should_module_be_updated"] = token
                res.append((kind, version, token, packet))
    for version in (0, 1, 2, 3):
        res.append(("metadata", version, "none", {"version": version}))
    assert len(res) == 52
    return res

def check(before, after, item):
    kind, version, token, packet = item
    old_ver, old_weights, old_opt, old_selector = before
    new_ver, new_weights, new_opt, new_selector = after
    module = "module" in packet
    optimizer = "optimizer" in packet
    selector = "should_module_be_updated" in packet
    accepted = module and (version == 0 or version > old_ver)
    expected_version = (
        old_ver + 1 if accepted and version == 0 else
        version if accepted else old_ver
    )
    expected_weights = token if accepted else old_weights
    expected_opt = (
        token if optimizer and (not module or accepted) else old_opt
    )
    expected_selector = (
        token if selector and (not module or accepted) else old_selector
    )
    expected = (
        expected_version, expected_weights, expected_opt, expected_selector
    )
    if after == expected:
        return None
    if new_ver != expected_version:
        return "weight_version_published_without_installation_or_regressed"
    if new_weights != expected_weights:
        return "module_admission_mismatch"
    if new_opt != expected_opt:
        return "optimizer_provenance_mismatch"
    if new_selector != expected_selector:
        return "trainability_selector_provenance_mismatch"
    raise AssertionError("Unclassified mismatch")

def run():
    cases = packets()
    count = 0
    failures = Counter()
    example = {}
    for length in (1, 2, 3):
        for trace in itertools.product(cases, repeat=length):
            obj = Learner()
            count += 1
            for step, p in enumerate(trace):
                before = obj.snapshot()
                set_state(obj, copy.deepcopy(p[3]))
                after = obj.snapshot()
                err = check(before, after, p)
                if err is not None:
                    failures[err] += 1
                    example.setdefault(err, (trace[:step+1], before, after))
                    break
    print(f"CHECKED traces={count} packet_types={len(cases)}, lengths=1,2,3")
    for err in sorted(failures):
        trace, old, new = example[err]
        print(
            f"COUNTEREXAMPLE {err} count={failures[err]} "
            f"first={[(p[0],p[1],p[2]) for p in trace]} before={old} after={new}"
        )
    if baseline:
        assert failures["trainability_selector_provenance_mismatch"] > 0, failures
        assert not {k:v for k,v in failures.items() if k != "trainability_selector_provenance_mismatch"}, failures
        print("PASS BASELINE: earlier paired-optimizer fix still violates trainability selector provenance")
    else:
        assert not failures, failures
        print("PASS PATCH: zero three-component causal-admission violations in finite model")
    print("LIMIT: bounded sequential calls; no exception rollback, intra-call races or full Ray cluster")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: checker.py LEARNER_SOURCE [--baseline]")
    run()
