"""Native Ray 2.59.0 Learner exception-atomicity experiment.

Use real installed Ray 2.59.0 and the *reviewed candidate* Learner.set_state
method copied/compiled from the checkout, not a mock Learner. Inject a
training-selector setter failure after module and optimizer and sequence
publication, then replay the same checkpoint. This experiment verifies a
transactional omission; it does not install a production solution.

Run outside the source checkout (cwd=/tmp), with the upstream source passed as
argv[1]. Requires ray[rllib]==2.59.0 and PyTorch CPU.
"""
from __future__ import annotations

import ast
import copy
import pathlib
import sys
from contextlib import suppress
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import ray
import ray.rllib.core.learner.learner as learner_module
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader

src = pathlib.Path(sys.argv[1])
tree = ast.parse(src.read_text(), filename=str(src))
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Learner")
f = copy.deepcopy(next(n for n in klass.body
                       if isinstance(n, ast.FunctionDef) and n.name == "set_state"))
f.decorator_list = []
f.returns = None
for a in [*f.args.args, *f.args.kwonlyargs]:
    a.annotation = None
env_ns = dict(vars(learner_module))
exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),
             str(src), "exec"), env_ns)
learner_module.Learner.set_state = env_ns["set_state"]


def float_param_snap(module_state):
    state = module_state[DEFAULT_MODULE_ID]
    return {k: np.array(v,copy=True) for k,v in state.items()
            if np.issubdtype(np.asarray(v).dtype,np.floating)}


def make_changed(module_state):
    out = copy.deepcopy(module_state)
    target=out[DEFAULT_MODULE_ID]
    modified=None
    for k,v in target.items():
        val=np.asarray(v)
        if val.size and np.issubdtype(val.dtype,np.floating):
            target[k]=np.asarray(v).copy()+np.asarray(.125,dtype=val.dtype)
            modified=k
            break
    assert modified is not None,"No floating RLModule parameter found"
    return out,modified


def changed(a,b):
    keys=a.keys() & b.keys()
    return max(float(np.max(np.abs(a[k]-b[k]))) for k in keys)


ray.init(num_cpus=2,include_dashboard=False,ignore_reinit_error=True,log_to_driver=False)
env = gym.make("CartPole-v1")
try:
    learner=BaseTestingAlgorithmConfig().build_learner(env=env)
    current=learner.get_state()
    # Force an accepted local weight install to establish a positive sequence.
    learner.set_state({COMPONENT_RL_MODULE:current[COMPONENT_RL_MODULE]})
    current=learner.get_state()
    version=int(current[WEIGHTS_SEQ_NO])
    assert version>0,version
    assert learner.should_module_be_updated(DEFAULT_MODULE_ID)
    original=float_param_snap(current[COMPONENT_RL_MODULE])

    mutated,key=make_changed(current[COMPONENT_RL_MODULE])
    packet={WEIGHTS_SEQ_NO:version+1,
            COMPONENT_RL_MODULE:mutated,
            COMPONENT_OPTIMIZER:copy.deepcopy(current[COMPONENT_OPTIMIZER]),
            "should_module_be_updated":{"definitely_not_default_module"}}
    def injection(**kwargs):
        raise RuntimeError("injected selector commit failure")

    with patch.object(learner.config,"multi_agent",side_effect=injection):
        try:
            learner.set_state(copy.deepcopy(packet))
        except RuntimeError as exc:
            assert str(exc)=="injected selector commit failure",str(exc)
        else:
            raise AssertionError("Selector failure did not trigger")

    after=learner.get_state()
    installed_params=float_param_snap(after[COMPONENT_RL_MODULE])
    delta=changed(original,installed_params)
    assert delta>0.08,delta
    assert after[WEIGHTS_SEQ_NO]==version+1,(version,after[WEIGHTS_SEQ_NO])
    assert learner.should_module_be_updated(DEFAULT_MODULE_ID)
    print("NATIVE WITNESS: selector fails after model installed, actual weights changed by",
          delta,"and version advanced to",after[WEIGHTS_SEQ_NO],flush=True)

    # Replaying this *same* valid checkpoint cannot repair the missing selector.
    learner.set_state(copy.deepcopy(packet))
    restored=learner.get_state()
    assert restored[WEIGHTS_SEQ_NO]==version+1
    assert learner.should_module_be_updated(DEFAULT_MODULE_ID),(
        "Equal-version retry unexpectedly applied previously rejected selector")
    print("NATIVE REPLAY: same version accepted as already installed; selector remains stale",
          flush=True)

    reader=get_cartpole_dataset_reader(batch_size=32)
    batch=reader.next().as_multi_agent()
    batch=learner._convert_batch_type(batch)
    before_training=float_param_snap(learner.get_state()[COMPONENT_RL_MODULE])
    result=learner.update(batch=batch)
    after_training=float_param_snap(learner.get_state()[COMPONENT_RL_MODULE])
    train_delta=changed(before_training,after_training)
    assert train_delta>0,train_delta
    print("NATIVE TRAINING CONSEQUENCE: default policy updated despite intended disabled "
          "selector; batch rows",batch.count,"max parameter delta",train_delta,
          "result keys",list(map(str,result)),flush=True)
    print("REAL RAY LEARNER PARTIAL-PUBLICATION COUNTEREXAMPLE CONFIRMED",flush=True)
finally:
    ray.shutdown()
    with suppress(Exception):
        env.close()
