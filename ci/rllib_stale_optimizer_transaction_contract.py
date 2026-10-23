"""Execute the real Learner.set_state source to audit paired versioned state replay.

No Ray installation or GPU required. This reproduces a transaction boundary
question; do not equate it to an observed RL training divergence.
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

path=Path(sys.argv[1]) if len(sys.argv)>1 and not sys.argv[1].startswith("--") else Path("python/ray/rllib/core/learner/learner.py")
baseline="--baseline" in sys.argv
tree=ast.parse(path.read_text(),filename=str(path))
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="Learner")
node=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="set_state")
node.decorator_list=[]
node.returns=None
node.args.args[1].annotation=None
scope={"COMPONENT_RL_MODULE":"module","COMPONENT_OPTIMIZER":"optimizer",
       "COMPONENT_METRICS_LOGGER":"metrics","WEIGHTS_SEQ_NO":"version"}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),"exec"),scope)
native_set_state=scope["set_state"]

class Module:
    def __init__(self):
        self.weights=20
    def set_state(self,state):
        self.weights=state["weights"]

class FakeLearner:
    def __init__(self):
        self.module=Module()
        self._weights_seq_no=10
        self.optimizer=20
        self.config=SimpleNamespace(multi_agent=lambda **kw:None)
        self.metrics=SimpleNamespace(set_state=lambda *_:None)
    def _check_is_built(self):
        pass
    def _set_optimizer_state(self,state):
        self.optimizer=state["momentum"]

# Genuine paired old-checkpoint replay, not an optimizer-only state update:
# A new v10 module and optimizer are live; a delayed versioned v9 message
# contains both v9 module weights and the corresponding v9 optimizer state.
learner=FakeLearner()
native_set_state(learner,{"version":9,"module":{"weights":9},
                          "optimizer":{"momentum":9}})
assert learner._weights_seq_no==10
assert learner.module.weights==20
if baseline:
    assert learner.optimizer==9,learner.optimizer
    print("EXPECTED BASELINE: module v10 retained but optimizer v9 applied")
else:
    assert learner.optimizer==20,learner.optimizer
    print("PASS PATCH: stale paired snapshot preserves both module and optimizer v10")

# Existing pull-request behaviors must remain intact.
unversioned=FakeLearner()
native_set_state(unversioned,{"module":{"weights":21},
                              "optimizer":{"momentum":21}})
assert (unversioned.module.weights,unversioned.optimizer,unversioned._weights_seq_no)==(21,21,11)
print("PASS unversioned module update bumps learner version")
fresh=FakeLearner()
native_set_state(fresh,{"version":12,"module":{"weights":12},
                        "optimizer":{"momentum":12}})
assert (fresh.module.weights,fresh.optimizer,fresh._weights_seq_no)==(12,12,12)
print("PASS newer versioned pair is restored consistently")
print("INTERPRETATION: consistency gap for stale paired snapshots is observed; intended restore semantics need maintainer confirmation")
