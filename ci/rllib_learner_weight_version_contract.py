"""Run REAL Ray RLlib Learner and EnvRunner set_state method bodies.

No Ray install or GPU required: extract method AST from checkout / pinned
upstream, replace only caller objects (modules, metrics, trainer config) with
deterministic scalar states. Tests causal consequences of version publication,
not production Ray actors, learner training, or throughput.
"""
import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

baseline = "--baseline" in sys.argv
paths = [Path(s) for s in sys.argv[1:] if not s.startswith("--")]
worker_path = paths[0] if paths else Path("python/ray/rllib/core/learner/learner.py")
runner_path = paths[1] if len(paths)>1 else Path("python/ray/rllib/env/multi_agent_env_runner.py")
WEIGHTS_SEQ_NO = "weights_seq_no"
COMPONENT_RL_MODULE = "rl_module"
COMPONENT_OPTIMIZER = "optimizer"
COMPONENT_METRICS_LOGGER = "metrics_logger"
COMPONENT_ENV_TO_MODULE_CONNECTOR = "env_to_module"
COMPONENT_MODULE_TO_ENV_CONNECTOR = "module_to_env"
NUM_ENV_STEPS_SAMPLED_LIFETIME = "num_env_steps_sampled_lifetime"

def method_from_source(path, classname, methodname):
    tree = ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==methodname)
    method.decorator_list=[]  # Only omit @override, not the method implementation.
    scope={
        "StateDict":dict,
        "WEIGHTS_SEQ_NO":WEIGHTS_SEQ_NO,
        "COMPONENT_RL_MODULE":COMPONENT_RL_MODULE,
        "COMPONENT_OPTIMIZER":COMPONENT_OPTIMIZER,
        "COMPONENT_METRICS_LOGGER":COMPONENT_METRICS_LOGGER,
        "COMPONENT_ENV_TO_MODULE_CONNECTOR":COMPONENT_ENV_TO_MODULE_CONNECTOR,
        "COMPONENT_MODULE_TO_ENV_CONNECTOR":COMPONENT_MODULE_TO_ENV_CONNECTOR,
        "NUM_ENV_STEPS_SAMPLED_LIFETIME":NUM_ENV_STEPS_SAMPLED_LIFETIME,
        "ray":SimpleNamespace(ObjectRef=type("ObjectRef",(),{}),get=lambda x:x),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
                 str(path),"exec"),scope)
    return scope[methodname]

learner_set_state=method_from_source(worker_path,"Learner","set_state")
runner_set_state=method_from_source(runner_path,"MultiAgentEnvRunner","set_state")

class Module:
    def __init__(self,weight=0):
        self.state={"policy":{"weight":weight}}
        self.updates=0
    def set_state(self,state):
        self.state=copy.deepcopy(state)
        self.updates+=1
    def get_state(self):
        return copy.deepcopy(self.state)

class Learner:
    set_state=learner_set_state
    def __init__(self,version=7):
        self.module=Module()
        self._weights_seq_no=version
        self.config=SimpleNamespace(multi_agent=lambda **kwargs:None)
        self.metrics=SimpleNamespace(set_state=lambda state:None)
    def _check_is_built(self):return True
    def _set_optimizer_state(self,state):pass

class Runner:
    set_state=runner_set_state
    def __init__(self,version=7):
        self.module=Module()
        self._weights_seq_no=version
        self.num_env_steps_sampled_lifetime=0

learner=Learner(version=7)
runner=Runner(version=7)
# This is the real mutation path used by LearnerGroup.set_weights(),
# including the implicit weight version that the learner publishes.
learner.set_state({COMPONENT_RL_MODULE:{"policy":{"weight":42}}})
assert learner.module.state["policy"]["weight"]==42
new_version=learner._weights_seq_no
runner.set_state({
  COMPONENT_RL_MODULE:learner.module.get_state(),
  WEIGHTS_SEQ_NO:new_version
})
if baseline:
    assert new_version==7
    assert runner.module.state["policy"]["weight"]==0
    assert runner._weights_seq_no==7
    print("EXPECTED BASELINE: Learner changes weight 0->42 but same seq=7; pull-if-newer EnvRunner ignores state")
else:
    assert new_version==8
    assert runner.module.state["policy"]["weight"]==42
    assert runner._weights_seq_no==8
    print("PASS: versioned propagation of mutated learner state to actual EnvRunner method")

# A delayed older checkpoint must not relabel a newer learner.
learner.set_state({
  COMPONENT_RL_MODULE:{"policy":{"weight":-1}},
  WEIGHTS_SEQ_NO:6
})
assert learner.module.state["policy"]["weight"]==42
if baseline:
    assert learner._weights_seq_no==6
    print("EXPECTED BASELINE: rejected stale version 6 overwrites current learner version 7")
else:
    assert learner._weights_seq_no==8
    print("PASS: rejected stale version 6 neither changes weights nor downgrades learner seq 8")

# A legitimate future update still works, and a version-less empty update
# must NOT invent a new weight mutation.
learner.set_state({WEIGHTS_SEQ_NO:10})
assert learner._weights_seq_no==10
learner.set_state({COMPONENT_OPTIMIZER:{"momentum":0.5}})
assert learner._weights_seq_no==10
print("PASS: future state versions and optimizer-only state retain their intended semantics")
