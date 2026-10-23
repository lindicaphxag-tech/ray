"""Test actual RLlib SelfPlayLeagueBasedCallback and EnvRunner methods on CPU.

This deterministic pre-merge audit executes on_train_result and
MultiAgentEnvRunner.set_state extracted from production source. No Ray service
or torch needed. In the unchanged prior PR revision, default sync_weights()
queues an asynchronous weight publish, so mapping may run first; the fix
forces initialization synchronously for every cloned opponent module.

The fake ActorManager exposes a valid worker-health schedule: remote 2 is
unavailable for the first dispatch and recovers before the second. The real
EnvRunnerGroup.foreach_env_runner source dispatches callbacks according to
that schedule. One install-and-map call prevents partial rollout activation.
"""
import ast
import copy
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from types import SimpleNamespace

baseline="--baseline" in sys.argv
paths=[Path(x) for x in sys.argv[1:] if not x.startswith("--")]
callback_path=paths[0] if paths else Path(
    "python/ray/rllib/examples/multi_agent/utils/self_play_league_based_callback.py")
runner_path=paths[1] if len(paths)>1 else Path(
    "python/ray/rllib/env/multi_agent_env_runner.py")
group_path=paths[2] if len(paths)>2 else Path("python/ray/rllib/env/env_runner_group.py")

class ObjectRef:
    def __init__(self, value): self.value=copy.deepcopy(value)

class Ray:
    ObjectRef=ObjectRef
    @staticmethod
    def put(value):return ObjectRef(value)
    @staticmethod
    def get(value):return copy.deepcopy(value.value)

def production_method(path, clsname, methodname, bindings):
    node=ast.parse(path.read_text(),filename=str(path))
    cls=next(x for x in node.body if isinstance(x,ast.ClassDef) and x.name==clsname)
    method=next(x for x in cls.body
                if isinstance(x,ast.FunctionDef) and x.name==methodname)
    method.decorator_list=[]
    env=dict(bindings)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
                 str(path),"exec"),env)
    return env[methodname]

runner_method=production_method(
    runner_path,"MultiAgentEnvRunner","set_state",
    {"StateDict":dict, "COMPONENT_RL_MODULE":"rl_module",
     "COMPONENT_ENV_TO_MODULE_CONNECTOR":"env_to_module",
     "COMPONENT_MODULE_TO_ENV_CONNECTOR":"module_to_env",
     "NUM_ENV_STEPS_SAMPLED_LIFETIME":"num_steps",
     "WEIGHTS_SEQ_NO":"weights_seq_no", "ray":Ray})

class Module:
    def __init__(self,weight):
        self.weight=weight
        self.installations=0
    def get_state(self):return {"weight":self.weight}
    def set_state(self,data):
        self.weight=data["weight"]
        self.installations+=1

class Bank(dict):
    def __init__(self,states,events=None,label=None):
        super().__init__(states)
        self.events=events
        self.label=label
    def get_state(self):
        return {k:v.get_state() for k,v in self.items()}
    def set_state(self,states):
        for k,value in states.items():
            self[k].set_state(value)
        if self.events is not None:
            self.events.append(("weights",self.label))

class Runner:
    set_state=runner_method
    def __init__(self,events,label):
        self._weights_seq_no=7
        self.module=Bank({
            "main":Module(42),"main_0":Module(-1),
            "league_exploiter_1":Module(-2),"main_exploiter_1":Module(-3)},
            events=events,label=label)
        self.events=events
        self.label=label
        self.mapped=False
        self.config=SimpleNamespace(multi_agent=self.install_mapping)
    def install_mapping(self,**kwargs):
        self.mapped=True
        self.events.append(("mapping",self.label))
        assert callable(kwargs["policy_mapping_fn"])

# Delegate synchronous dispatch to the REAL EnvRunnerGroup.foreach_env_runner.
class RemoteResult:
    ok=True
    def __init__(self,value):self.value=value
    def get(self):return self.value

class RemoteResults(list):
    def ignore_errors(self):return self

class Manager:
    def __init__(self,remotes):
        self.remotes=remotes
        self.calls=0
    def actor_ids(self):return [1,2]
    def foreach_actor(self,func,**kwargs):
        assert kwargs["timeout_seconds"] is None, "The synchronous path must not time out"
        assert kwargs["healthy_only"] is True
        self.calls+=1
        # The second worker becomes healthy between separate RPC rounds.
        health=[1] if self.calls==1 else [1,2]
        return RemoteResults([RemoteResult(func(self.remotes[i])) for i in health])

group_foreach=production_method(
    group_path,"EnvRunnerGroup","foreach_env_runner",
    {"Union":Union,"Callable":Callable,"List":List,"Dict":Dict,"Optional":Optional,
     "Any":Any,"EnvRunner":Runner,"T":object,
     "FaultTolerantActorManager":SimpleNamespace(
         handle_remote_call_result_errors=lambda *args,**kwargs:None)})

class Group:
    foreach_env_runner=group_foreach
    def __init__(self,local_runner,remotes,events):
        self.local_env_runner=local_runner
        self._worker_manager=Manager(remotes)
        self._ignore_ray_errors_on_env_runners=False
        self._num_env_runners_dropped_lifetime=0
        self.events=events
        self.queued=[]
    def sync_weights(self,**kwargs):
        # The original default dispatch is async: a newly enqueued weight
        # packet need not have finished when the next mapping call runs.
        self.queued.append(kwargs)
        self.events.append(("async_publish",None))

class LearnerGroup:
    def __init__(self):self.states=[]
    def set_weights(self,states):self.states.append(copy.deepcopy(states))
    def foreach_learner(self,*args,**kwargs):return []

cb_method=production_method(
    callback_path,"SelfPlayLeagueBasedCallback","on_train_result",
    {"re":re,"np":SimpleNamespace(random=SimpleNamespace(choice=lambda x,**k:x[0])),
     "ENV_RUNNER_RESULTS":"env_runner_results",
     "COMPONENT_RL_MODULE":"rl_module",
     "RLModuleSpec":SimpleNamespace(from_module=lambda m:m),
     "ray":Ray})

class Callback:
    on_train_result=cb_method
    def __init__(self):
        self.trainable_policies={"main"}
        self.non_trainable_policies={"main_0","main_exploiter_0","league_exploiter_0"}
        self.main_policies={"main","main_0"}
        self.main_exploiters={"main_exploiter_0","main_exploiter_1"}
        self.league_exploiters={"league_exploiter_0","league_exploiter_1"}
        self.win_rate_threshold=0.5
        self.win_rates={}
    def _print_league(self):pass

local=Bank({
    "main":Module(42),"main_0":Module(0),
    "league_exploiter_1":Module(0),"main_exploiter_1":Module(0)})
events=[]
local_runner=Runner(events,"local")
remote1=Runner(events,"remote1")
remote2=Runner(events,"remote2")
group=Group(local_runner,{1:remote1,2:remote2},events)
learner_group=LearnerGroup()
algorithm=SimpleNamespace(iteration=1,env_runner=SimpleNamespace(module=local),
    env_runner_group=group,learner_group=learner_group)
callback=Callback()
callback.on_train_result(algorithm=algorithm,
    result={"env_runner_results":{"win_rate_main":0.9}})
assert len(learner_group.states)==1
assert set(callback.trainable_policies)=={
    "main","league_exploiter_1","main_exploiter_1"}
ids=("main_0","league_exploiter_1","main_exploiter_1")
assert all(local[mid].weight==42 for mid in ids)
assert not group.queued,group.queued
ids=("main_0","league_exploiter_1","main_exploiter_1")
for mid in ids:
    assert local[mid].weight==42
if baseline:
    assert group._worker_manager.calls==2
    assert events==[
        ("weights","local"),("weights","remote1"),
        ("mapping","local"),("mapping","remote1"),
        ("mapping","remote2")], events
    assert remote2.mapped
    assert any(remote2.module[mid].weight!=42 for mid in ids)
    print("EXPECTED BASELINE: remote2 recovers between RPCs and receives mapping without cloned weights")
else:
    assert group._worker_manager.calls==1
    assert events==[
        ("weights","local"),("mapping","local"),
        ("weights","remote1"),("mapping","remote1")],events
    for r in (local_runner,remote1):
        assert r.mapped
        assert all(r.module[mid].weight==42 for mid in ids)
        assert all(r.module[mid].installations==1 for mid in ids)
    assert not remote2.mapped
    assert all(remote2.module[mid].weight!=42 for mid in ids)
    print("PASS: per-runner single RPC prevents activation without weights on recovering runner")
print("PASS: executed real Callback / EnvRunnerGroup / EnvRunner source under changing health sets")
