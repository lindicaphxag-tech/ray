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
import ray
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE
from ray.rllib.core.learner import learner as learner_module
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO

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
            setter.assert_not_called()
        assert learner.get_state()[WEIGHTS_SEQ_NO]==seq
        print(f"PASS native Learner rejected whole paired snapshot at version {incoming}")

    # An optimizer-only component with future version must not claim that
    # new module weights were installed.
    state=learner.get_state()
    learner.set_state({
        WEIGHTS_SEQ_NO:seq+1,
        COMPONENT_OPTIMIZER:state[COMPONENT_OPTIMIZER],
    })
    assert learner.get_state()[WEIGHTS_SEQ_NO]==seq
    print("PASS native Learner optimizer-only component leaves weight version unchanged")

    learner.set_state({
        WEIGHTS_SEQ_NO:seq+1,
        COMPONENT_RL_MODULE:state[COMPONENT_RL_MODULE],
    })
    assert learner.get_state()[WEIGHTS_SEQ_NO]==seq+1
    print("PASS native Learner accepts module weights at next published version")

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
    print("PASS native Learner restores module and optimizer together when newer")
finally:
    ray.shutdown()
    env.close()
print("PASS live installed-wheel RLlib Learner state contract smoke")
