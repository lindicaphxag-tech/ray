"""Real same-version *different provenance* Learner state collision.

Two independent Adam training histories can legitimately have the same
WEIGHTS_SEQ_NO. An incoming paired state at that version must not silently
apply optimizer moments without its corresponding parameters.

The fixture runs the ACTUAL production Learner.set_state method on real
PyTorch Adam parameters/optimizer moments. This is a diagnostic of an
unresolved consistency condition, not a proposed native API behavior.
"""
from __future__ import annotations
import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

path=Path(sys.argv[1]) if len(sys.argv)>1 and not sys.argv[1].startswith("--") else Path("python/ray/rllib/core/learner/learner.py")
baseline="--baseline" in sys.argv
tree=ast.parse(path.read_text(),filename=str(path))
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="Learner")
node=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="set_state"))
node.decorator_list=[]
node.returns=None
node.args.args[1].annotation=None
namespace={"COMPONENT_RL_MODULE":"module","COMPONENT_OPTIMIZER":"optimizer",
           "COMPONENT_METRICS_LOGGER":"metrics","WEIGHTS_SEQ_NO":"version"}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),"exec"),namespace)
native_set_state=namespace["set_state"]

class Module:
    def __init__(self,param):self.param=param
    def set_state(self,state):
        with torch.no_grad():self.param.copy_(state["parameter"])

class Learner:
    def __init__(self):
        self.parameter=torch.nn.Parameter(torch.tensor([0.5,-1.25],dtype=torch.float64))
        self.module=Module(self.parameter)
        self.optimizer=torch.optim.Adam([self.parameter],lr=.03)
        self._weights_seq_no=0
        self.config=SimpleNamespace(multi_agent=lambda **kw:None)
        self.metrics=SimpleNamespace(set_state=lambda *_:None)
    def _check_is_built(self):pass
    def _set_optimizer_state(self,state):
        self.optimizer.load_state_dict(copy.deepcopy(state))
    def step(self,target):
        self.optimizer.zero_grad()
        loss=((self.parameter-torch.as_tensor(target,dtype=torch.float64))**2).mean()
        loss.backward()
        self.optimizer.step()
        self._weights_seq_no+=1
    def snapshot(self):
        return {"version":self._weights_seq_no,
                "module":{"parameter":self.parameter.detach().clone()},
                "optimizer":copy.deepcopy(self.optimizer.state_dict())}

local=Learner()
remote=Learner()
for i in range(10):
    local.step([1.0+i*.05,-0.4+i*.02])
    remote.step([-1.2-i*.04,2.0+i*.08])
assert local._weights_seq_no==remote._weights_seq_no==10
local_snap=local.snapshot()
remote_snap=remote.snapshot()
assert not torch.equal(local_snap["module"]["parameter"],remote_snap["module"]["parameter"])
assert not torch.equal(local_snap["optimizer"]["state"][0]["exp_avg"],remote_snap["optimizer"]["state"][0]["exp_avg"])
assert local_snap["optimizer"]["state"][0]["step"].item()==remote_snap["optimizer"]["state"][0]["step"].item()==10

# Equal version, but distinct *legitimate* optimizer histories.
native_set_state(local,remote_snap)
actual_params=local.parameter.detach()
actual_moments=local.optimizer.state_dict()["state"][0]["exp_avg"]
assert torch.equal(actual_params,local_snap["module"]["parameter"])
if baseline:
    assert torch.equal(actual_moments,remote_snap["optimizer"]["state"][0]["exp_avg"])
    print("EXPECTED BASELINE: equal seq=10 skips module but loads unrelated remote Adam moments")
else:
    assert torch.equal(actual_moments,local_snap["optimizer"]["state"][0]["exp_avg"])
    print("PASS PATCH: equal seq=10 preserves both local module and local Adam moments")

# Construct a matched reference and quantify the update discrepancy.
matched=Learner()
with torch.no_grad():matched.parameter.copy_(local_snap["module"]["parameter"])
matched.optimizer.load_state_dict(copy.deepcopy(local_snap["optimizer"]))
matched._weights_seq_no=10
local.step([2.6,.7])
matched.step([2.6,.7])
delta=float(torch.linalg.vector_norm(local.parameter-matched.parameter))
if baseline:
    assert delta>1e-5,delta
    print(f"EXPECTED BASELINE: same-version partial restore changes next Adam update; L2={delta:.12f}")
else:
    assert delta<1e-12,delta
    print(f"PASS PATCH: same-version complete rejection preserves exact next update (L2={delta:.12f})")
print("INTERPRETATION: version labels are not unique state identities across independent learners")
