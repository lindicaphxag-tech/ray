"""Test actual RLlib SelfPlayLeagueBasedCallback and EnvRunner methods on CPU.

This deterministic pre-merge audit executes on_train_result and
MultiAgentEnvRunner.set_state extracted from production source. No Ray service
or torch needed. In the unchanged prior PR revision, default sync_weights()
queues an asynchronous weight publish, so mapping may run first; the fix
forces initialization synchronously for every cloned opponent module.

The fake EnvRunnerGroup captures a valid schedule where the async publish
has not completed when mapping is set. The actual production EnvRunner
set_state method applies the patched module-only state.
"""
import ast
import copy
import re
import sys
from pathlib import Path
from types import SimpleNamespace

baseline="--baseline" in sys.argv
paths=[Path(x) for x in sys.argv[1:] if not x.startswith("--")]
callback_path=paths[0] if paths else Path(
    "python/ray/rllib/examples/multi_agent/utils/self_play_league_based_callback.py")
runner_path=paths[1] if len(paths)>1 else Path(
    "python/ray/rllib/env/multi_agent_env_runner.py")

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
    def get_state(self):
        return {k:v.get_state() for k,v in self.items()}
    def set_state(self,states):
        for k,value in states.items():
            self[k].set_state(value)

class Runner:
    set_state=runner_method
    def __init__(self,events):
        self._weights_seq_no=7
        self.module=Bank({
            "main":Module(42),"main_0":Module(-1),
            "league_exploiter_1":Module(-2),"main_exploiter_1":Module(-3)})
        self.events=events
        self.config=SimpleNamespace(multi_agent=self.install_mapping)
    def install_mapping(self,**kwargs):
        self.events.append(("mapping",set(kwargs["policies_to_train"])))
        assert callable(kwargs["policy_mapping_fn"])

class Group:
    def __init__(self,runner,events):
        self.runner=runner
        self.events=events
        self.queued=[]
    def foreach_env_runner(self,func,**kwargs):
        assert kwargs.get("local_env_runner") is True
        before=len(self.events)
        result=func(self.runner)
        # The real config callback logs mapping directly. A module-only state
        # callback does not, so capture its successful installation here.
        if len(self.events)==before:
            self.events.append(("weight_installed",self.runner.module.get_state()))
        else:
            assert len(self.events)==before+1
            assert self.events[-1][0]=="mapping"
        return [result]
    def sync_weights(self,**kwargs):
        # A legal execution schedule of RLlib's async-fetch-ready path: the
        # new publish is queued but has not executed before mapping.
        self.queued.append(kwargs)
        self.events.append(("async_publish",kwargs))

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
remote=Runner(events)
group=Group(remote,events)
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
assert len([x for x in events if x[0]=="mapping"])==1,events
if baseline:
    assert len(group.queued)==1 and group.queued[0].get("policies")==list(ids)
    assert all(remote.module[mid].weight!=42 for mid in ids)
    assert [x[0] for x in events][:2]==["async_publish","mapping"],events
    print("EXPECTED BASELINE: first league mapping installed before its async cloned weights")
else:
    assert not group.queued
    assert all(remote.module[mid].weight==42 for mid in ids)
    assert all(remote.module[mid].installations==1 for mid in ids)
    assert events[0][0]=="weight_installed" and events[1][0]=="mapping",events
    assert remote._weights_seq_no==7
    print("PASS: initial three self-play clones installed before mapping, version gate bypassed")
print("PASS: actual callback / EnvRunner source-version contract")
