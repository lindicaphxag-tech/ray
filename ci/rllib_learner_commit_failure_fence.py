"""Ray Learner native production-method fault injection: partial publication.

Execute the actual reviewed three-component Learner.set_state function body via
AST, injecting exceptions at the component setters. Assert torn state and
same-version replay rejection explicitly. No Ray distributed runtime, no
claim of transactional fix or universally novel protocol.

This is a negative/diagnostic experiment against a proposed fork review patch,
NOT a test of an upstream merged version.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

PATH = Path("python/ray/rllib/core/learner/learner.py")
tree = ast.parse(PATH.read_text(encoding="utf-8"), filename=str(PATH))
klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Learner")
method = copy.deepcopy(next(x for x in klass.body
                            if isinstance(x, ast.FunctionDef) and x.name == "set_state"))
method.decorator_list = []
method.returns = None
for arg in method.args.args:
    arg.annotation = None
names = {
    "COMPONENT_RL_MODULE": "rl_module",
    "COMPONENT_OPTIMIZER": "optimizer",
    "COMPONENT_METRICS_LOGGER": "metrics",
    "WEIGHTS_SEQ_NO": "weights_seq_no",
}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
             str(PATH), "exec"), names)
native_set_state = names["set_state"]


class InjectedFailure(RuntimeError):
    pass


class Module:
    def __init__(self, parent):
        self.parent = parent

    def set_state(self, state):
        self.parent.trigger("module:before")
        self.parent.weights = copy.deepcopy(state["identity"])
        self.parent.trace.append("module:wrote")
        self.parent.trigger("module:after")


class Config:
    def __init__(self, parent):
        self.parent = parent
        self.policies_to_train = {"live_policy"}

    def multi_agent(self, *, policies_to_train):
        self.parent.trigger("selector:before")
        self.policies_to_train = set(policies_to_train)
        self.parent.trace.append("selector:wrote")
        self.parent.trigger("selector:after")


class Metrics:
    def set_state(self, state):
        return None


class Learner:
    set_state = native_set_state

    def __init__(self, fail_at=None):
        self._weights_seq_no = 10
        self.weights = "live:w10"
        self.optimizer = "live:adam10"
        self.trace = []
        self.fail_at = fail_at
        self.module = Module(self)
        self.config = Config(self)
        self.metrics = Metrics()

    def _check_is_built(self):
        return None

    def trigger(self, point):
        self.trace.append(point)
        if point == self.fail_at:
            raise InjectedFailure(point)

    def _set_optimizer_state(self, state):
        self.trigger("optimizer:before")
        self.optimizer = copy.deepcopy(state["identity"])
        self.trace.append("optimizer:wrote")
        self.trigger("optimizer:after")

    def snapshot(self):
        return (self._weights_seq_no, self.weights, self.optimizer,
                tuple(sorted(self.config.policies_to_train)))


NEW = {
    "weights_seq_no": 11,
    "rl_module": {"identity": "foreign:w11"},
    "optimizer": {"identity": "foreign:adam11"},
    "should_module_be_updated": {"foreign_policy"},
}
OLD = (10, "live:w10", "live:adam10", ("live_policy",))
FULL = (11, "foreign:w11", "foreign:adam11", ("foreign_policy",))


def validate_actual_method():
    clean = Learner()
    clean.set_state(copy.deepcopy(NEW))
    assert clean.snapshot() == FULL, clean.snapshot()
    print("PASS clean production set_state: all three components and v11 accepted",flush=True)

    outcomes = {}
    for stage in (
        "module:before", "module:after", "optimizer:before",
        "optimizer:after", "selector:before", "selector:after"
    ):
        learner = Learner(fail_at=stage)
        try:
            learner.set_state(copy.deepcopy(NEW))
            raise AssertionError(f"{stage} did not raise")
        except InjectedFailure as exc:
            assert str(exc) == stage
        before = learner.snapshot()
        outcomes[stage] = before
        print("INJECTED",stage,"live=",before,"trace=",learner.trace,flush=True)
        # Verify that failed calls are NOT generally all-or-nothing.
        if stage != "module:before":
            assert before != OLD, (stage,before)
        # The selector can throw after the module+optimizer+version have already
        # been published; the checkpoint's selector is NOT installed yet.
        if stage == "selector:before":
            assert before == (11,"foreign:w11","foreign:adam11",("live_policy",))
            learner.fail_at = None
            learner.set_state(copy.deepcopy(NEW))
            assert learner.snapshot() == before, "same-version replay unexpectedly repaired torn selector"
            print("CERTIFIED selector-before failure publishes v11, but same-version replay is a no-op",flush=True)
        if stage == "optimizer:after":
            assert before == (10,"foreign:w11","foreign:adam11",("live_policy",)),before
            print("CERTIFIED optimizer failure writes two components without publishing version",flush=True)

    assert outcomes["module:before"] == OLD
    assert outcomes["selector:after"] == FULL
    assert any(v not in (OLD,FULL) for v in outcomes.values())
    return outcomes


class FailClosedGate:
    """Illustrative wrapper; not a native Learner patch and not crash durable."""
    def __init__(self, learner):
        self.learner = learner
        self.uncertain = False

    def set_state(self, packet):
        if self.uncertain:
            raise RuntimeError("checkpoint publication uncertain: rebuild Learner")
        try:
            self.learner.set_state(packet)
        except BaseException:
            self.uncertain = True
            raise

    def train(self):
        if self.uncertain:
            raise RuntimeError("training prevented until authoritative rebuild")
        return self.learner.snapshot()


def demonstrate_fail_closed():
    for point in ["module:after","optimizer:after","selector:before","selector:after"]:
        obj = Learner(fail_at=point)
        gate = FailClosedGate(obj)
        try:
            gate.set_state(copy.deepcopy(NEW))
        except InjectedFailure:
            pass
        else:
            raise AssertionError("injected failure was not raised")
        assert gate.uncertain
        for fn in [gate.train,lambda:gate.set_state(copy.deepcopy(NEW))]:
            try:
                fn()
            except RuntimeError as err:
                assert "uncertain" in str(err) or "prevented" in str(err)
            else:
                raise AssertionError("unsafe operation proceeded after failed publication")
        print("FAIL_CLOSED demonstrator quarantined after",point,flush=True)


if __name__ == "__main__":
    cases = validate_actual_method()
    demonstrate_fail_closed()
    print("SOURCE-EXACT COMMIT-FAILURE DIAGNOSTIC PASSED, cases=",len(cases),flush=True)
