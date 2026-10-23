"""Live RLlib Learner smoke with the candidate production set_state method.

Keeps every Ray module other than the single Learner.set_state method from the
same installed Ray wheel. The *actual current fork production method body* is
compiled from an explicit source path and installed on the wheel Learner class.
This checks runtime method/API integration, not full git-master Ray builds.
"""
from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE
from ray.rllib.core.learner import learner as learner_module
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO

baseline="--baseline" in sys.argv
source=Path(sys.argv[1])
tree=ast.parse(source.read_text(), filename=str(source))
cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="Learner")
fn=copy.deepcopy(next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="set_state"))
fn.decorator_list=[]
fn.returns=None
fn.args.args[1].annotation=None
scope=dict(vars(learner_module))
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),"exec"),scope)
learner_module.Learner.set_state=scope["set_state"]

print("Ray wheel version:",ray.__version__)
print("Native Learner wheel:",learner_module.__file__)
print("Patched method loaded from:",source)

env=gym.make("CartPole-v1")
ray.init(num_cpus=2,ignore_reinit_error=True,include_dashboard=False,log_to_driver=False)
try:
    learner=BaseTestingAlgorithmConfig().build_learner(env=env)
    initial=learner.get_state()
    learner.set_state({COMPONENT_RL_MODULE:initial[COMPONENT_RL_MODULE]})
    learner.set_state({COMPONENT_RL_MODULE:initial[COMPONENT_RL_MODULE]})
    seq=learner.get_state()[WEIGHTS_SEQ_NO]
    assert seq>=2,seq

    # Versioned stale and equal model-plus-optimizer packets must not
    # partially install optimizer state while refusing module weights.
    for incoming in (seq-1,seq):
        state=learner.get_state()
        with mock.patch.object(learner,"_set_optimizer_state",
                               wraps=learner._set_optimizer_state) as setter:
            learner.set_state({
                WEIGHTS_SEQ_NO:incoming,
                COMPONENT_RL_MODULE:state[COMPONENT_RL_MODULE],
                COMPONENT_OPTIMIZER:state[COMPONENT_OPTIMIZER],
            })
            if baseline:
                setter.assert_called_once()
                print(f"EXPECTED BASELINE native Learner partially accepts version {incoming}")
            else:
                setter.assert_not_called()
                print(f"PASS PATCH native Learner rejects whole paired version {incoming}")
        assert learner.get_state()[WEIGHTS_SEQ_NO]==seq

    # An optimizer-only component with future version must not claim that
    # new module weights were installed. Mutate a real float tensor to detect
    # whether the subsequent matching module version is actually accepted.
    state=learner.get_state()
    future_module=copy.deepcopy(state[COMPONENT_RL_MODULE])
    changed=None
    for module_id,params in future_module.items():
        for key,value in params.items():
            if isinstance(value,np.ndarray) and np.issubdtype(value.dtype,np.floating):
                params[key]=value+1
                changed=(module_id,key)
                break
        if changed is not None:break
    assert changed is not None
    learner.set_state({
        WEIGHTS_SEQ_NO:seq+1,
        COMPONENT_OPTIMIZER:state[COMPONENT_OPTIMIZER],
    })
    if baseline:
        assert learner.get_state()[WEIGHTS_SEQ_NO]==seq+1
        print("EXPECTED BASELINE: optimizer-only transfer falsely publishes unseen weight version")
    else:
        assert learner.get_state()[WEIGHTS_SEQ_NO]==seq
        print("PASS PATCH: optimizer-only transfer leaves weight version unchanged")

    learner.set_state({
        WEIGHTS_SEQ_NO:seq+1,
        COMPONENT_RL_MODULE:future_module,
    })
    after=learner.get_state()
    assert after[WEIGHTS_SEQ_NO]==seq+1
    module_id,key=changed
    if baseline:
        assert np.array_equal(after[COMPONENT_RL_MODULE][module_id][key],
                              state[COMPONENT_RL_MODULE][module_id][key])
        print("EXPECTED BASELINE: true v-next module discarded")
    else:
        assert np.array_equal(after[COMPONENT_RL_MODULE][module_id][key],
                              future_module[module_id][key])
        print("PASS PATCH: true v-next module installed")

    state=learner.get_state()
    with mock.patch.object(learner,"_set_optimizer_state",
                           wraps=learner._set_optimizer_state) as setter:
        learner.set_state({
            WEIGHTS_SEQ_NO:seq+2,
            COMPONENT_RL_MODULE:state[COMPONENT_RL_MODULE],
            COMPONENT_OPTIMIZER:state[COMPONENT_OPTIMIZER],
        })
        setter.assert_called_once()
    assert learner.get_state()[WEIGHTS_SEQ_NO]==seq+2
    print("PASS: native Learner restores module and optimizer together when newer")
finally:
    ray.shutdown()
    env.close()
print("PASS live installed-wheel RLlib Learner state contract smoke")
