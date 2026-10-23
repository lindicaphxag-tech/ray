"""Causal state-publication proof: optimizer-only state must not install weight-version claims.

Uses the actual Ray Learner.set_state method body with real torch.nn.Parameter
and Adam. Source is extracted rather than imitated, preserving production
branch ordering. An optimizer-only packet with a newer weight version
cannot make the Learner reject the matching future module weights.

This is a source-contract probe with a deterministic two-parameter model,
not an end-to-end distributed Ray Learner integration run.
"""
from __future__ import annotations
import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

baseline="--baseline" in sys.argv
path=Path(sys.argv[1]) if len(sys.argv)>1 and not sys.argv[1].startswith("--") else Path("python/ray/rllib/core/learner/learner.py")
tree=ast.parse(path.read_text(),filename=str(path))
cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="Learner")
fn=copy.deepcopy(next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=="set_state"))
fn.decorator_list=[]
fn.returns=None
fn.args.args[1].annotation=None
scope={
    "COMPONENT_RL_MODULE":"module",
    "COMPONENT_OPTIMIZER":"optimizer",
    "COMPONENT_METRICS_LOGGER":"metrics",
    "WEIGHTS_SEQ_NO":"version",
}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),"exec"),scope)
native_set_state=scope["set_state"]

# Document the real API's ability to export versioned optimizer-only state.
getstate=next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=="get_state")
getstate_source=ast.unparse(getstate)
assert "WEIGHTS_SEQ_NO" in getstate_source
assert "COMPONENT_OPTIMIZER" in getstate_source

class Module:
    def __init__(self,param): self.param=param
    def set_state(self,state):
        with torch.no_grad(): self.param.copy_(state["parameter"])

class Learner:
    def __init__(self):
        self.parameter=torch.nn.Parameter(torch.tensor([1.0,-1.0],dtype=torch.float64))
        self.module=Module(self.parameter)
        self.optimizer=torch.optim.Adam([self.parameter],lr=.03)
        self._weights_seq_no=10
        self.config=SimpleNamespace(multi_agent=lambda **kw:None)
        self.metrics=SimpleNamespace(set_state=lambda *_:None)
    def _check_is_built(self): pass
    def _set_optimizer_state(self,state):
        self.optimizer.load_state_dict(copy.deepcopy(state))
    def step(self,target):
        self.optimizer.zero_grad()
        ((self.parameter-torch.tensor(target,dtype=torch.float64))**2).mean().backward()
        self.optimizer.step()
        self._weights_seq_no+=1
    def snapshot(self):
        return {"version":self._weights_seq_no,
                "module":{"parameter":self.parameter.detach().clone()},
                "optimizer":copy.deepcopy(self.optimizer.state_dict())}

# Fresh v11 source contains *matching* optimizer and model state. Its optimizer
# component can arrive first via a component-scoped state transfer.
producer=Learner()
producer.step([2.0,0.3])
future=producer.snapshot()
assert future["version"]==11
local=Learner()
assert local._weights_seq_no==10
local_before=local.parameter.detach().clone()
optim_only={"version":11,"optimizer":copy.deepcopy(future["optimizer"])}

native_set_state(local,optim_only)
assert torch.equal(local.parameter,local_before)
if baseline:
    assert local._weights_seq_no==11
    print("EXPECTED BASELINE: optimizer-only transfer advertised unseen model v11 without applying its weights")
else:
    assert local._weights_seq_no==10
    print("PASS PATCH: optimizer-only transfer does not claim uninstalled model version 11")

native_set_state(local,future)
if baseline:
    assert local._weights_seq_no==11
    assert torch.equal(local.parameter,local_before)
    assert not torch.equal(local.parameter,future["module"]["parameter"])
    print("EXPECTED BASELINE: true v11 module rejected after optimizer-only fake publication")
else:
    assert local._weights_seq_no==11
    assert torch.equal(local.parameter,future["module"]["parameter"])
    print("PASS PATCH: subsequent real module v11 accepted after optimizer-only transfer")

print("CAUSAL INVARIANT: weight version may advance only on a module write or actual training update")
