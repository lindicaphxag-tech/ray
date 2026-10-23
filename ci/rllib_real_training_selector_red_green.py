"""One real Ray RLlib Learner update after rejected trainable-selector state.

Executes the actual Learner.set_state from the current Ray source against
an installed compatible Ray 2.59.0 runtime. Tests whether rejected snapshot
metadata only mutates a config key, or actually changes the subsequent
gradient-based update for the default policy.

This is a single CPU training-step correctness regression, not an RL
performance comparison or proof of general convergence.
"""
from __future__ import annotations
import ast
import copy
import pathlib
import sys
from contextlib import suppress

import gymnasium as gym
import numpy as np
import ray
import ray.rllib.core.learner.learner as learner_module
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import (
    BaseTestingAlgorithmConfig,
    get_cartpole_dataset_reader,
)
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO

if len(sys.argv) not in (2, 3):
    raise SystemExit("usage: actual_training.py candidate_learner.py [--baseline]")
source = pathlib.Path(sys.argv[1])
baseline = "--baseline" in sys.argv

tree = ast.parse(source.read_text(), filename=str(source))
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Learner")
method = copy.deepcopy(next(node for node in cls.body
                           if isinstance(node, ast.FunctionDef) and node.name == "set_state"))
method.decorator_list = []
method.returns = None
method.args.args[1].annotation = None
scope = dict(vars(learner_module))
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
             str(source), "exec"), scope)
learner_module.Learner.set_state = scope["set_state"]

env = gym.make("CartPole-v1")
ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
try:
    learner = BaseTestingAlgorithmConfig().build_learner(env=env)
    initial = learner.get_state()
    learner.set_state({COMPONENT_RL_MODULE: initial[COMPONENT_RL_MODULE]})
    installed = learner.get_state()
    version = installed[WEIGHTS_SEQ_NO]
    assert version > 0
    assert learner.should_module_be_updated(DEFAULT_MODULE_ID)

    # Rejected equal-version model state from a different training context:
    # the original setter *still* applies its selector.
    learner.set_state({
        WEIGHTS_SEQ_NO: version,
        COMPONENT_RL_MODULE: installed[COMPONENT_RL_MODULE],
        COMPONENT_OPTIMIZER: installed[COMPONENT_OPTIMIZER],
        "should_module_be_updated": {"stale_opponent"},
    })
    should_train = learner.should_module_be_updated(DEFAULT_MODULE_ID)
    if baseline:
        assert not should_train, "original setter did not apply stale selector"
        print("EXPECTED BASELINE: default policy became untrainable after rejected snapshot")
    else:
        assert should_train, "patch allowed rejected selector to disable live policy"
        print("PASS PATCH: default policy remains eligible for gradient updates")

    before = learner.get_state()[COMPONENT_RL_MODULE][DEFAULT_MODULE_ID]
    before = {key: np.asarray(value).copy() for key, value in before.items()}
    reader = get_cartpole_dataset_reader(batch_size=32)
    batch = reader.next().as_multi_agent()
    batch = learner._convert_batch_type(batch)
    print("Training batch rows:",batch.count)
    assert batch.count > 0
    # The actual installed Ray Learner.update() filters non-trainable modules.
    try:
        results = learner.update(batch=batch)
        print("Actual learner update result keys:", sorted(map(str, results.keys())))
    except (ValueError, RuntimeError, KeyError) as exc:
        if not baseline:
            raise
        print("EXPECTED BASELINE: update cannot run after stale selector:", str(exc)[:200])

    after = learner.get_state()[COMPONENT_RL_MODULE][DEFAULT_MODULE_ID]
    differences = [
        float(np.max(np.abs(np.asarray(after[key]) - before[key])))
        for key in before
        if np.issubdtype(np.asarray(before[key]).dtype, np.floating)
    ]
    assert differences, "No floating model parameters found"
    max_param_change = max(differences)
    print("Actual max absolute parameter update:", format(max_param_change, ".10g"))

    if baseline:
        assert max_param_change == 0, max_param_change
        print("EXPECTED BASELINE: stale selector prevented real gradient update")
    else:
        assert max_param_change > 0, max_param_change
        print("PASS PATCH: real default-policy parameters updated on actual training batch")
finally:
    ray.shutdown()
    with suppress(Exception):
        env.close()
