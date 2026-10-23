"""Real Ray Actor restore-RPC audit for dynamic policy routing.

Executes source-extracted Algorithm.restore_env_runners and
MultiAgentEnvRunner.get_state/set_state from pinned source with a real
FaultTolerantActorManager and a remote worker actor. Explicitly exercises
restoration into a freshly constructed remote actor. Does NOT simulate actor
process crashes, real environment episodes, or training.
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import ray
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.actor_manager import FaultTolerantActorManager

baseline="--baseline" in sys.argv
paths=[Path(x) for x in sys.argv[1:] if not x.startswith("--")]
alg_path=paths[0] if paths else Path("python/ray/rllib/algorithms/algorithm.py")
runner_path=paths[1] if len(paths)>1 else Path("python/ray/rllib/env/multi_agent_env_runner.py")

def method(path,clsname,name,ns):
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==clsname)
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)
    fn.decorator_list=[]
    for x in ast.walk(fn):
        if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)):
            x.returns=None
            for y in [*x.args.posonlyargs,*x.args.args,*x.args.kwonlyargs]:
                y.annotation=None
            if x.args.vararg:x.args.vararg.annotation=None
            if x.args.kwarg:x.args.kwarg.annotation=None
    space=ns.copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                 str(path),"exec"),space)
    return space[name]

class M:
    def __init__(self,weight):self.weight=weight
    def get_state(self,**kw):return {"weight":self.weight}
    def set_state(self,s):self.weight=s["weight"]

class Bank(dict):
    def __init__(self,items,events):
        super().__init__(items)
        self.events=events
    @property
    def _rl_modules(self):return self
    def get_state(self,**kw):return {k:v.get_state() for k,v in self.items()}
    def set_state(self,state):
        for mid,weights in state.items():
            self[mid].set_state(weights)
        self.events.append("weights")
    def add_module(self,mid,module,override=False):self[mid]=module
    def remove_module(self,mid,raise_err_if_not_found=False):del self[mid]

class Spec:
    def __init__(self,w):self.w=w
    def build(self):return M(-999)
class MultiSpec:
    def __init__(self,modules):
        self.rl_module_specs={k:Spec(v.weight) for k,v in modules.items()}
    @staticmethod
    def from_module(modules):return MultiSpec(modules)

class Config:
    enable_env_runner_and_connector_v2=True
    is_multi_agent=True
    def __init__(self,mapping,trainable,events):
        self.policy_mapping_fn=lambda *a,**kw:mapping
        self.policies_to_train=set(trainable)
        self.events=events
    def multi_agent(self,**kw):
        if "policy_mapping_fn" in kw:self.policy_mapping_fn=kw["policy_mapping_fn"]
        if "policies_to_train" in kw:self.policies_to_train=set(kw["policies_to_train"])
        self.events.append("mapping")

class Connector:
    def get_state(self):return {"ready":True}
    def set_state(self,x):assert x=={"ready":True}

names={"ray":ray,"COMPONENT_RL_MODULE":"rl_module",
       "COMPONENT_ENV_TO_MODULE_CONNECTOR":"env_to_module",
       "COMPONENT_MODULE_TO_ENV_CONNECTOR":"module_to_env",
       "NUM_ENV_STEPS_SAMPLED_LIFETIME":"steps","WEIGHTS_SEQ_NO":"weights_seq_no"}
actual_get=method(runner_path,"MultiAgentEnvRunner","get_state",names)
actual_set=method(runner_path,"MultiAgentEnvRunner","set_state",names)

class Runner:
    get_state=actual_get
    set_state=actual_set
    def __init__(self,restored=False):
        self.events=[]
        self.module=Bank(
            {"main":M(-7)} if restored else {"main":M(42),"league_1":M(17)},
            self.events)
        self.config=Config("main" if restored else "league_1",
            {"main"} if restored else {"main","league_1"},self.events)
        self._weights_seq_no=0 if restored else 5
        self.num_env_steps_sampled_lifetime=19
        self._env_to_module=Connector()
        self._module_to_env=Connector()
    def _check_component(self,*a):return True
    def _get_subcomponents(self,*a):return None
    def report(self):
        return {"weights":self.module.get_state(),
            "seq":self._weights_seq_no,
            "route":self.config.policy_mapping_fn(),
            "trainable":sorted(self.config.policies_to_train),
            "events":self.events}

@ray.remote(num_cpus=.3)
class Remote(Runner):
    def __init__(self):super().__init__(restored=True)
    def apply(self,func):return func(self)
    def get_report(self):return self.report()

algo_ns={"ray":ray,"MultiRLModuleSpec":MultiSpec,
    "COMPONENT_LEARNER":"learner","COMPONENT_RL_MODULE":"rl_module",
    "COMPONENT_ENV_TO_MODULE_CONNECTOR":"env_to_module",
    "COMPONENT_MODULE_TO_ENV_CONNECTOR":"module_to_env",
    "NUM_ENV_STEPS_SAMPLED_LIFETIME":"steps","ENV_RUNNER_RESULTS":"runner_results"}
actual_restore=method(alg_path,"Algorithm","restore_env_runners",algo_ns)
class Algorithm:
    restore_env_runners=actual_restore
    def __init__(self,local):
        self.config=SimpleNamespace(is_online=True,
            enable_env_runner_and_connector_v2=True,is_multi_agent=True,
            env_runner_restore_timeout_s=None)
        self._counters=defaultdict(int)
        self.env_runner=local

class Group:
    def __init__(self,local,manager):
        self.local_env_runner=local
        self._worker_manager=manager
        self._ignore_ray_errors_on_env_runners=False
        self._num_env_runners_dropped_lifetime=0
    def probe_unhealthy_env_runners(self):
        # Controlled reconstruction is already complete, so activate the
        # normal restoration method for this now healthy new actor.
        return self._worker_manager.actor_ids()
    def foreach_env_runner(self,func,**kwargs):
        return EnvRunnerGroup.foreach_env_runner(self,func,**kwargs)

def main():
    ray.init(num_cpus=3,include_dashboard=False,logging_level="ERROR")
    try:
        local=Runner()
        worker=Remote.remote()
        manager=FaultTolerantActorManager([worker])
        group=Group(local,manager)
        algo=Algorithm(local)
        assert algo.restore_env_runners(group)==manager.actor_ids()
        report=ray.get(worker.get_report.remote())
        assert report["weights"]=={"main":{"weight":42},"league_1":{"weight":17}}
        assert report["seq"]==5
        if baseline:
            assert report["route"]=="main"
            assert report["trainable"]==["main"]
            assert report["events"]==["weights"],report
            print("EXPECTED BASELINE: real restored Ray actor has new policy weights but stale policy map",report)
        else:
            assert report["route"]=="league_1"
            assert report["trainable"]==["league_1","main"]
            assert report["events"]==["weights","mapping"],report
            print("PASS: real restored Ray actor adopts source weights then dynamic routing and trainable set",report)
        print("PASS: actual production restore methods delivered through native Ray actor RPC")
    finally:
        ray.shutdown()

if __name__=="__main__":main()
