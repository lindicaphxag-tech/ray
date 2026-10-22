"""Real MultiAgentEnvRunner.set_state source red/green version invariant.

CPU only. Exercises the exact production method body in the proposed branch
and an unchanged pinned upstream branch, with a deterministic module object.
It does not claim real network reordering or native Ray integration testing.
"""
import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

baseline = "--baseline" in sys.argv
path = Path(next((x for x in sys.argv[1:] if not x.startswith("--")),
    "python/ray/rllib/env/multi_agent_env_runner.py"))
root = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
cls = next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name=="MultiAgentEnvRunner")
method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="set_state")
method.decorator_list=[]  # Remove @override only; execute unchanged method body.
scope={
    "StateDict":dict,
    "COMPONENT_RL_MODULE":"rl_module",
    "WEIGHTS_SEQ_NO":"weights_seq_no",
    "COMPONENT_ENV_TO_MODULE_CONNECTOR":"env_to_module_connector",
    "COMPONENT_MODULE_TO_ENV_CONNECTOR":"module_to_env_connector",
    "NUM_ENV_STEPS_SAMPLED_LIFETIME":"steps",
    "ray":SimpleNamespace(ObjectRef=type("ObjectRef",(),{}),get=lambda x:x),
}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
             str(path),"exec"),scope)
method=scope["set_state"]

class Module:
    def __init__(self):
        self.state={"policy":{"weight":8}}
        self.applied=[]
    def set_state(self,state):
        self.state=copy.deepcopy(state)
        self.applied.append(copy.deepcopy(state))

class Runner:
    set_state=method
    def __init__(self):
        self._weights_seq_no=8
        self.module=Module()
        self.num_env_steps_sampled_lifetime=0

runner=Runner()
runner.set_state({"rl_module":{"policy":{"weight":-7}},"weights_seq_no":7})
assert runner.module.state=={"policy":{"weight":8}}
assert runner.module.applied==[]
if baseline:
    assert runner._weights_seq_no==7
    print("EXPECTED BASELINE: reject stale v7 weights, but downgrade resident version 8->7")
else:
    assert runner._weights_seq_no==8
    print("PASS: stale v7 packet cannot overwrite version-8 weights or version")

# Re-deliver the same already resident version. Baseline's false downgrade
# triggers a redundant module install; the patch preserves the equal-version gate.
runner.set_state({"rl_module":{"policy":{"weight":-8}},"weights_seq_no":8})
if baseline:
    assert runner._weights_seq_no==8
    assert runner.module.state["policy"]["weight"]==-8
    assert len(runner.module.applied)==1
    print("EXPECTED BASELINE: spurious redundant version-8 install follows a stale v7 packet")
else:
    assert runner._weights_seq_no==8
    assert runner.module.state["policy"]["weight"]==8
    assert runner.module.applied==[]
    print("PASS: equal-version delivery is a no-op")

# A legitimate newer version is still applied.
runner.set_state({"rl_module":{"policy":{"weight":9}},"weights_seq_no":9})
assert runner._weights_seq_no==9
assert runner.module.state["policy"]["weight"]==9
assert len(runner.module.applied)==(2 if baseline else 1)
print("PASS: newer v9 data and version published together")

# Module-only versionless messages retain their explicit force-install
# contract, but must not synthesize a newer global sequence.
runner.set_state({"rl_module":{"policy":{"weight":100}}})
assert runner.module.state["policy"]["weight"]==100
assert runner._weights_seq_no==9
print("PASS: forced module-only install keeps global sequence at v9")
