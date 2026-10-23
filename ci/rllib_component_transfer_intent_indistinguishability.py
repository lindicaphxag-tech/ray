"""An indistinguishable-intent counterexample in RLlib component-scoped state.

This script compiles and executes *actual* Learner.get_state and Learner.set_state
production methods. When only the optimizer is selected, get_state still exports
trainability selector metadata. Therefore, the exact same wire packet may mean
(A) "restore optimizer and trainability" or (B) "restore optimizer only, selector
was incidental metadata". Without an intent/provenance field, a deterministic
receiver cannot honor both semantics from that identical packet.

This is a necessary-information counterexample, not a claim of new original
information theory, or a proof that component-scoped metadata is a bug.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace


filename = "python/ray/rllib/core/learner/learner.py"
tree = ast.parse(Path(filename).read_text(), filename=filename)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Learner")
methods = []
for name in ("get_state", "set_state"):
    node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    node.returns = None
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        arg.annotation = None
    if node.args.kwarg: node.args.kwarg.annotation = None
    if node.args.vararg: node.args.vararg.annotation = None
    methods.append(node)

symbols = {
    "COMPONENT_RL_MODULE": "rl_module",
    "COMPONENT_OPTIMIZER": "optimizer",
    "COMPONENT_METRICS_LOGGER": "metrics",
    "WEIGHTS_SEQ_NO": "weights_seq_no",
}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
             filename, "exec"), symbols)


class Module:
    def __init__(self, parent):
        self.parent=parent
    def get_state(self, **kwargs):
        return {"identity": self.parent.weights}
    def set_state(self, state):
        self.parent.weights=state["identity"]

class Config:
    def __init__(self, selector):
        self.policies_to_train=set(selector)
    def multi_agent(self, *, policies_to_train):
        self.policies_to_train=set(policies_to_train)

class Learner:
    get_state = symbols["get_state"]
    set_state = symbols["set_state"]
    def __init__(self, origin, seq, selector):
        self.weights=f"{origin}:weights:{seq}"
        self.optimizer=f"{origin}:optimizer:{seq}"
        self._weights_seq_no=seq
        self.module=Module(self)
        self.config=Config(selector)
        self.metrics=SimpleNamespace(get_state=lambda: {},set_state=lambda x:None)
    def _check_is_built(self):
        return None
    def _check_component(self, kind, components, not_components):
        if not_components is not None:
            return kind not in (not_components if isinstance(not_components,(list,tuple,set)) else {not_components})
        return components is None or kind in (
            components if isinstance(components,(list,tuple,set)) else {components}
        )
    def _get_subcomponents(self, *args):
        return None
    def _get_optimizer_state(self):
        return {"identity":self.optimizer}
    def _set_optimizer_state(self, state):
        self.optimizer=state["identity"]

remote = Learner("remote", 9, {"opponent"})
local = Learner("local", 10, {"main"})
wire_packet = remote.get_state(components="optimizer")
assert set(wire_packet)=={"weights_seq_no","optimizer","should_module_be_updated"},wire_packet
assert wire_packet["weights_seq_no"]==9
assert wire_packet["should_module_be_updated"]=={"opponent"}
assert wire_packet["optimizer"]=={"identity":"remote:optimizer:9"}
print("PASS native get_state(components=optimizer) emitted optimizer, trainable selector AND weight version")
assert "rl_module" not in wire_packet

# This is allowed by the compatibility-preserving current candidate: absent
# module => optimizer and trainable selector are independently applied, and
# weight version cannot move.
local.set_state(wire_packet)
assert local._weights_seq_no==10
assert local.weights=="local:weights:10"
assert local.optimizer=="remote:optimizer:9"
assert local.config.policies_to_train=={"opponent"}
print("OBSERVED native setter accepts incidental selector on optimizer-only scoped state")

# Two possible caller intentions lead to exactly the same bytes:
# (A) restore optimizer and trainability; (B) restore only optimizer,
# but native get_state automatically attached trainability metadata.
# The receiving code sees identical (S,packet), so deterministic policy cannot
# both apply and reject the selector without additional caller-intent info.
desired_A={"opponent"}
desired_B={"main"}
assert desired_A!=desired_B
assert wire_packet==remote.get_state(components="optimizer")
print("PASS indistinguishable-intent certificate: same state+packet, incompatible desired selectors")
print("IMPLICATION additional explicit transfer intent or attested component binding is necessary")
print("SCOPE this proves ambiguity, not which existing API behavior maintainers should choose")
