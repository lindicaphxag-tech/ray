"""Causal learner-state review: stale checkpoints can silently rewrite training policy selection.

Executes actual checked-out production Learner.set_state (extracted AST) rather
than an imitated setter. The two output states contain module weights,
optimizer moments, and policy-training selector, all from distinct producers.
The checker verifies paired admission, selector-only updates, and unversioned
restores. This is a deterministic CPU unit boundary, not a full RL experiment.
"""
from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

baseline = "--baseline" in sys.argv
source = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path(
    "python/ray/rllib/core/learner/learner.py"
)
tree = ast.parse(source.read_text(), filename=str(source))
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Learner")
method = copy.deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "set_state"))
method.decorator_list = []
method.args.args[1].annotation = None
method.returns = None
symbols = {
    "COMPONENT_RL_MODULE": "rl_module",
    "COMPONENT_OPTIMIZER": "optimizer",
    "COMPONENT_METRICS_LOGGER": "metrics_logger",
    "WEIGHTS_SEQ_NO": "weights_seq_no",
}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
             str(source), "exec"), symbols)
native_set_state = symbols["set_state"]

get_state = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "get_state")
assert any(isinstance(n, ast.Constant) and n.value == "should_module_be_updated" for n in ast.walk(get_state))
print("PASS source audit: actual get_state exports should_module_be_updated alongside version")

class Module:
    def __init__(self, learner):
        self.learner = learner
    def set_state(self, state):
        self.learner.weights = state["origin"]

class Config:
    def __init__(self):
        self.policies_to_train = {"main"}
        self.calls = []
    def multi_agent(self, *, policies_to_train):
        self.policies_to_train = set(policies_to_train)
        self.calls.append(self.policies_to_train)

class Learner:
    def __init__(self):
        self._weights_seq_no = 10
        self.weights = "A:v10"
        self.optimizer = "A:v10"
        self.module = Module(self)
        self.config = Config()
        self.metrics = SimpleNamespace(set_state=lambda *_:None)
    def _check_is_built(self):
        return None
    def _set_optimizer_state(self, state):
        self.optimizer = state["origin"]

def state(learner):
    return (learner.weights, learner.optimizer, learner._weights_seq_no,
            frozenset(learner.config.policies_to_train))

for version in (9, 10):
    subject = Learner()
    packet = {
        "weights_seq_no": version,
        "rl_module": {"origin": f"B:v{version}"},
        "optimizer": {"origin": f"B:v{version}"},
        "should_module_be_updated": {"opponent"},
    }
    native_set_state(subject, packet)
    if baseline:
        assert state(subject) == ("A:v10", "A:v10", 10, frozenset({"opponent"})), state(subject)
        print(f"EXPECTED BASELINE: rejected paired v{version} yet overwrote training selector")
    else:
        assert state(subject) == ("A:v10", "A:v10", 10, frozenset({"main"})), state(subject)
        print(f"PASS PATCH: rejected paired v{version} preserves original trainable set")

# Selector is part of the training *task*. Show the next hypothetical update
# would route to a different module under the mismatched selector.
a = Learner()
native_set_state(a, {
    "weights_seq_no": 9,
    "rl_module": {"origin": "B:v9"},
    "optimizer": {"origin": "B:v9"},
    "should_module_be_updated": {"opponent"},
})
training_route = [p for p in ("main", "opponent") if p in a.config.policies_to_train]
if baseline:
    assert training_route == ["opponent"]
    print("EXPECTED BASELINE: next module-training selection flips to stale opponent")
else:
    assert training_route == ["main"]
    print("PASS PATCH: training selection preserves live module eligibility")

fresh = Learner()
native_set_state(fresh, {
    "weights_seq_no": 11,
    "rl_module": {"origin": "B:v11"},
    "optimizer": {"origin": "B:v11"},
    "should_module_be_updated": {"opponent"},
})
assert state(fresh) == ("B:v11", "B:v11", 11, frozenset({"opponent"}))
print("PASS newly versioned paired state installs weights, optimizer and selector")

selector_only = Learner()
native_set_state(selector_only, {
    "weights_seq_no": 13,
    "should_module_be_updated": {"opponent"},
})
assert state(selector_only) == ("A:v10", "A:v10", 10, frozenset({"opponent"}))
print("PASS selector-only update remains independent, does not claim new weight version")

module_only = Learner()
native_set_state(module_only, {
    "weights_seq_no": 10,
    "rl_module": {"origin": "B:v10"},
    "should_module_be_updated": {"opponent"},
})
if baseline:
    assert state(module_only) == ("A:v10", "A:v10", 10, frozenset({"opponent"}))
    print("EXPECTED BASELINE: equal-version module rejected while selector still changes")
else:
    assert state(module_only) == ("A:v10", "A:v10", 10, frozenset({"main"}))
    print("PASS PATCH: equal-version module and paired selector jointly rejected")

unversioned = Learner()
native_set_state(unversioned, {
    "rl_module": {"origin": "C:v0"},
    "optimizer": {"origin": "C:v0"},
    "should_module_be_updated": {"opponent"},
})
assert state(unversioned) == ("C:v0", "C:v0", 11, frozenset({"opponent"}))
print("PASS unversioned checkpoint can force-install all three components")
