"""Execute REAL RLlib Algorithm.restore_env_runners and EnvRunner state methods.

Runs unchanged upstream source and candidate source through identical
deterministic recovery. This is a source-extracted CPU regression, not a
running Ray cluster, and proves only the local-source restore path.
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

baseline = "--baseline" in sys.argv
sources = [Path(x) for x in sys.argv[1:] if not x.startswith("--")]
alg_path = sources[0] if sources else Path("python/ray/rllib/algorithms/algorithm.py")
runner_path = sources[1] if len(sources)>1 else Path("python/ray/rllib/env/multi_agent_env_runner.py")

class Ref:
    def __init__(self, obj):self.value=obj
class Ray:
    ObjectRef=Ref
    @staticmethod
    def put(x):return Ref(x)
    @staticmethod
    def get(x):return x.value

def extract_method(path, classname, methodname, names):
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==classname)
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==methodname)
    fn.decorator_list=[]
    # Avoid importing the heavy framework solely to evaluate type annotations.
    for n in ast.walk(fn):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            n.returns=None
            for a in [*n.args.posonlyargs,*n.args.args,*n.args.kwonlyargs]:
                a.annotation=None
            if n.args.vararg:n.args.vararg.annotation=None
            if n.args.kwarg:n.args.kwarg.annotation=None
    ns=names.copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                 str(path),"exec"),ns)
    return ns[methodname]

class M:
    def __init__(self,w):self.weight=w
    def get_state(self,**kw):return {"weight":self.weight}
    def set_state(self,s):self.weight=s["weight"]

events=[]
class Bank(dict):
    @property
    def _rl_modules(self):return self
    def get_state(self,**kw):return {k:v.get_state() for k,v in self.items()}
    def set_state(self,states):
        for mid,s in states.items():self[mid].set_state(s)
        events.append("weights")
    def add_module(self,mid,module,override=False):self[mid]=module
    def remove_module(self,mid,raise_err_if_not_found=False):del self[mid]

class Spec:
    def __init__(self,w):self.w=w
    def build(self):return M(-1)
class MultiSpec:
    def __init__(self,module):
        self.rl_module_specs={mid:Spec(m.weight) for mid,m in module.items()}
    @staticmethod
    def from_module(module):return MultiSpec(module)

class Config:
    enable_env_runner_and_connector_v2=True
    is_multi_agent=True
    def __init__(self,routing,trainable):
        self.policy_mapping_fn=lambda *a,**kw:routing
        self.policies_to_train=set(trainable)
    def multi_agent(self,**kwargs):
        if "policy_mapping_fn" in kwargs:
            self.policy_mapping_fn=kwargs["policy_mapping_fn"]
        if "policies_to_train" in kwargs:
            self.policies_to_train=set(kwargs["policies_to_train"])
        events.append("mapping")

class Connector:
    def get_state(self):return {"status":True}
    def set_state(self,state):assert state=={"status":True}

runner_globs={
    "ray":Ray,
    "COMPONENT_RL_MODULE":"rl_module",
    "COMPONENT_ENV_TO_MODULE_CONNECTOR":"env_to_module",
    "COMPONENT_MODULE_TO_ENV_CONNECTOR":"module_to_env",
    "NUM_ENV_STEPS_SAMPLED_LIFETIME":"steps",
    "WEIGHTS_SEQ_NO":"weights_seq_no",
}
get_state=extract_method(runner_path,"MultiAgentEnvRunner","get_state",runner_globs)
set_state=extract_method(runner_path,"MultiAgentEnvRunner","set_state",runner_globs)
class Runner:
    get_state=get_state
    set_state=set_state
    def __init__(self,module,config,version):
        self.module=module
        self.config=config
        self._weights_seq_no=version
        self.num_env_steps_sampled_lifetime=10
        self._env_to_module=Connector()
        self._module_to_env=Connector()
    def _check_component(self,*a):return True
    def _get_subcomponents(self,*a):return None

local=Runner(Bank({"main":M(42),"league_1":M(17)}),
             Config("league_1",{"main","league_1"}),5)
recovered=Runner(Bank({"main":M(-1)}),Config("main",{"main"}),0)
assert "policy_mapping_fn" not in local.get_state()
assert "policies_to_train" not in local.get_state()

alg_globs={
    "ray":Ray,
    "MultiRLModuleSpec":MultiSpec,
    "COMPONENT_RL_MODULE":"rl_module",
    "COMPONENT_LEARNER":"learner",
    "COMPONENT_ENV_TO_MODULE_CONNECTOR":"env_to_module",
    "COMPONENT_MODULE_TO_ENV_CONNECTOR":"module_to_env",
    "NUM_ENV_STEPS_SAMPLED_LIFETIME":"steps",
    "ENV_RUNNER_RESULTS":"runner_results",
}
restore=extract_method(alg_path,"Algorithm","restore_env_runners",alg_globs)
class Group:
    local_env_runner=local
    def probe_unhealthy_env_runners(self):return [2]
    def foreach_env_runner(self,func,**kwargs):
        assert kwargs["remote_worker_ids"]==[2]
        assert kwargs["local_env_runner"] is False
        func(recovered)
        return [None]
class Algo:
    restore_env_runners=restore
    def __init__(self):
        self.config=SimpleNamespace(is_online=True,
            enable_env_runner_and_connector_v2=True,
            is_multi_agent=True,env_runner_restore_timeout_s=None)
        self.env_runner=local
        self._counters=defaultdict(int)
        self.learner_group=None
algo=Algo()
restored=algo.restore_env_runners(Group())
assert restored==[2]
assert algo._counters["total_num_restored_workers"]==1
assert recovered.module["main"].weight==42
assert recovered.module["league_1"].weight==17
assert recovered._weights_seq_no==5
if baseline:
    assert recovered.config.policy_mapping_fn()=="main"
    assert recovered.config.policies_to_train=={"main"}
    assert events==["weights"],events
    print("EXPECTED BASELINE: restored weights/modules/seq, but stale original policy route and trainable set")
else:
    assert recovered.config.policy_mapping_fn()=="league_1"
    assert recovered.config.policies_to_train=={"main","league_1"}
    assert events==["weights","mapping"],events
    print("PASS: actual restore and get/set state methods restore module weights before dynamic mapping/training config")
print("PASS source-extracted recovery invariants, original vs candidate")
