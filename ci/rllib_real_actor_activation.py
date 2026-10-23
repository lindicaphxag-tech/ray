"""Real Ray actor RPC regression for per-runner self-play activation.

Uses the released Ray wheel's compiled _raylet, real ray.remote actors,
FaultTolerantActorManager, EnvRunnerGroup.foreach_env_runner, and the
checked-out production SelfPlayLeagueBasedCallback.on_train_result.
Fake policy modules have scalar weights for deterministic correctness checks;
no RL training or genuine fault recovery is claimed. Actor health is
deliberately toggled between the two dispatches of the old implementation.
"""
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import ray
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.examples.multi_agent.utils.self_play_league_based_callback import SelfPlayLeagueBasedCallback
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

baseline = "--baseline" in sys.argv
if baseline:
    candidate = next(x for x in sys.argv[1:] if not x.startswith("--"))
    spec=importlib.util.spec_from_file_location("unchanged_previous_callback",candidate)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Callback=module.SelfPlayLeagueBasedCallback
else:
    Callback=SelfPlayLeagueBasedCallback

class Module:
    def __init__(self,weight):
        self.weight=weight
        self.installs=0
    def get_state(self):
        return {"weight":self.weight}
    def set_state(self,state):
        self.weight=state["weight"]
        self.installs+=1

class Bank(dict):
    def get_state(self):
        return {name:mod.get_state() for name,mod in self.items()}
    def set_state(self,states):
        for mid,st in states.items():
            self[mid].set_state(st)

class Config:
    def __init__(self,owner):
        self.owner=owner
    def multi_agent(self,**kwargs):
        self.owner.mapping_enabled=True
        self.owner.trainable=sorted(kwargs["policies_to_train"])
        # No actor should activate a policy using its default weights.
        self.owner.activation_weights={
            mid:self.owner.module[mid].weight for mid in
            ("main_0","league_exploiter_1","main_exploiter_1")
        }

class Worker:
    def __init__(self,label):
        self.label=label
        self.module=Bank({
            "main":Module(42),
            "main_0":Module(-1),
            "league_exploiter_1":Module(-2),
            "main_exploiter_1":Module(-3),
        })
        self._weights_seq_no=7
        self.config=Config(self)
        self.mapping_enabled=False
        self.activation_weights=None
        self.trainable=None
        self.num_env_steps_sampled_lifetime=0
    def set_state(self,state):
        return MultiAgentEnvRunner.set_state(self,state)
    def report(self):
        return {"name":self.label,"mapped":self.mapping_enabled,
            "weights":self.module.get_state(),"installed":{
                name:mod.installs for name,mod in self.module.items()},
            "activation_weights":self.activation_weights,
            "version":self._weights_seq_no}

@ray.remote(num_cpus=0.3)
class Remote(Worker):
    def apply(self,func):
        return func(self)
    def get_report(self):
        return self.report()

class HealthChangeManager:
    def __init__(self,manager):
        self.inner=manager
        self.calls=0
    def actor_ids(self):
        return self.inner.actor_ids()
    def foreach_actor(self,func,**kw):
        self.calls+=1
        if self.calls==2:
            self.inner.set_actor_state(self.inner.actor_ids()[-1],True)
        return self.inner.foreach_actor(func,**kw)

class EnvRunnerDispatch:
    def __init__(self,group):
        self.group=group
    def foreach_env_runner(self,func,**kwargs):
        return EnvRunnerGroup.foreach_env_runner(self.group,func,**kwargs)
    def sync_weights(self,**kwargs):
        raise AssertionError("The tested PR has no async sync_weights call")

def check():
    ray.init(num_cpus=3,include_dashboard=False,logging_level="ERROR")
    try:
        a1=Remote.remote("remote1")
        a2=Remote.remote("remote2")
        inner=FaultTolerantActorManager(
            [a1,a2], max_remote_requests_in_flight_per_actor=2)
        inner.set_actor_state(inner.actor_ids()[-1],False)
        manager=HealthChangeManager(inner)
        local=Worker("local")
        dispatch=SimpleNamespace(
            local_env_runner=local,
            _worker_manager=manager,
            _ignore_ray_errors_on_env_runners=False,
            _num_env_runners_dropped_lifetime=0,
        )
        source=Bank({
            "main":Module(42),"main_0":Module(0),
            "league_exploiter_1":Module(0),"main_exploiter_1":Module(0)})
        learner=MagicMock()
        algorithm=SimpleNamespace(
            iteration=1,
            env_runner=SimpleNamespace(module=source),
            env_runner_group=EnvRunnerDispatch(dispatch),
            learner_group=learner,
        )
        callback=Callback(win_rate_threshold=.5)
        callback.on_train_result(
            algorithm=algorithm,
            result={ENV_RUNNER_RESULTS:{"win_rate_main":.9}})
        reports=[local.report(),ray.get(a1.get_report.remote()),
                 ray.get(a2.get_report.remote())]
        assert all(source[mid].weight==42 for mid in
                   ("main_0","league_exploiter_1","main_exploiter_1"))
        assert learner.set_weights.call_count==1
        assert all(r["version"]==7 for r in reports)
        assert all(r["mapped"] and
                   all(v==42 for v in r["activation_weights"].values())
                   for r in reports[:2]),reports
        if baseline:
            assert manager.calls==2,manager.calls
            assert reports[2]["mapped"], reports
            assert any(v!=42 for v in reports[2]["activation_weights"].values()),reports
            print("EXPECTED BASELINE: real remote2 actor received mapping without first receiving its clone weights",reports)
        else:
            assert manager.calls==1,manager.calls
            assert not reports[2]["mapped"],reports
            assert all(r["installed"]["main_0"]==1 for r in reports[:2])
            print("PASS: real Ray actors install all initial clone weights before mapping",reports)
        print("PASS: real actor RPC & actual ActorManager health gate")
    finally:
        ray.shutdown()

if __name__=="__main__":
    check()
