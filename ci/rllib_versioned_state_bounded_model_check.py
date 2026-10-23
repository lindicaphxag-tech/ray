"""Bounded exhaustive message-order audit of the actual RLlib Learner.set_state.

All sequences of length 1..3 from 25 incoming state packets are evaluated.
Real production method is AST extracted from a pinned source file; the model
and optimizer holders are instrumented doubles for fast exhaustive checking.
This is bounded *model checking*, NOT a proof of all possible Ray states,
nor an atomic-rollback or interleaving-within-method guarantee.
"""
from __future__ import annotations

import ast
import copy
import itertools
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def production_method(source):
    tree = ast.parse(Path(source).read_text(), filename=source)
    cl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Learner")
    fn = copy.deepcopy(next(n for n in cl.body if isinstance(n, ast.FunctionDef) and n.name == "set_state"))
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
                 source, "exec"), scope)
    return scope["set_state"]


class Model:
    def __init__(self, owner):
        self.owner = owner
    def set_state(self, state):
        self.owner.weights = state["identity"]


class Learner:
    def __init__(self):
        self._weights_seq_no = 1
        self.weights = "local:1"
        self.optimizer = "local:1"
        self.module = Model(self)
        self.metrics = SimpleNamespace(set_state=lambda *_: None)
        self.config = SimpleNamespace(multi_agent=lambda **kw: None)
    def _check_is_built(self):
        pass
    def _set_optimizer_state(self, state):
        self.optimizer = state["identity"]


def packets():
    result = []
    for kind in ("pair", "module", "optimizer"):
        for version in (0, 1, 2, 3):
            for source in ("A", "B"):
                token = f"{source}:{version}"
                value = {"version": version}
                if kind in ("pair", "module"):
                    value["module"] = {"identity": token}
                if kind in ("pair", "optimizer"):
                    value["optimizer"] = {"identity": token}
                result.append((kind, version, token, value))
    for version in (1, 2, 3):
        result.append(("metadata", version, "none", {"version": version}))
    return result


def check_step(before, after, p):
    kind, version, token, _ = p
    old_v, old_w, old_opt = before
    new_v, new_w, new_opt = after
    has_module = kind in ("pair", "module")
    if not has_module and new_v != old_v:
        return "false_weight_publication"
    if kind == "pair" and 0 < version <= old_v:
        if after != before:
            return "partial_paired_admission"
    if kind == "pair" and version > old_v:
        if (new_w, new_opt, new_v) != (token, token, version):
            return "new_paired_restore_failed"
    if kind == "module" and version > old_v:
        if (new_w, new_v) != (token, version):
            return "new_weight_delivery_failed"
    if has_module and version == 0 and new_v != old_v + 1:
        return "unversioned_weight_not_published"
    if new_v < old_v:
        return "weight_version_regressed"
    return None


def run(path, baseline):
    method = production_method(path)
    candidates = packets()
    assert len(candidates) == 27
    totals = Counter()
    first = {}
    evaluated = 0
    for length in (1, 2, 3):
        for trace in itertools.product(candidates, repeat=length):
            learner = Learner()
            for step, p in enumerate(trace):
                before = (learner._weights_seq_no, learner.weights, learner.optimizer)
                method(learner, copy.deepcopy(p[3]))
                after = (learner._weights_seq_no, learner.weights, learner.optimizer)
                violation = check_step(before, after, p)
                if violation:
                    totals[violation] += 1
                    first.setdefault(violation, (trace[:step+1], before, after))
                    break
            evaluated += 1
    print(f"CHECKED {evaluated} ordered traces of length 1..3, packet_types={len(candidates)}")
    for kind in sorted(totals):
        trace, before, after = first[kind]
        short = [(p[0], p[1], p[2]) for p in trace]
        print(f"COUNTEREXAMPLE {kind}: {totals[kind]} traces; first={short}; before={before}; after={after}")
    if baseline:
        assert totals["partial_paired_admission"] > 0, totals
        assert totals["false_weight_publication"] > 0, totals
        print("PASS BASELINE: unmodified source violates both independently specified causal invariants")
    else:
        assert not totals, totals
        print("PASS PATCH: no causal-admission/weight-publication violation in bounded exhaustive search")
    print("LIMITATION: models method-call order, NOT simultaneous mutation or exception rollback")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: checker.py PATH [--baseline]")
    run(sys.argv[1], "--baseline" in sys.argv)
