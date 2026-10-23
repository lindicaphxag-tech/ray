"""Run the *real checked-out native TestLearner methods* on a coherent Ray wheel.

Extract only target unittest methods from the fork test_learner.py AST, rather
than copying an entire development test module with incompatible wheel imports.
In the same Python runtime load the actual candidate/baseline Learner.set_state
body from a supplied source path into the installed Ray Learner class.

This is stronger than a hand-built smoke test (native checked-out test bodies
are executed), but is NOT a complete master-source native test environment.
"""
from __future__ import annotations

import ast
import copy
import pathlib
import sys
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np
import ray
import ray.rllib.core.learner.learner as learner_module
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO

if len(sys.argv) not in (3, 4):
    raise SystemExit("usage: native_methods.py native_test_file learner_source [--baseline]")
source_file = pathlib.Path(sys.argv[1])
learner_file = pathlib.Path(sys.argv[2])
baseline = len(sys.argv) == 4 and sys.argv[3] == "--baseline"

# Avoid version-specific full source module import (Ray 2.59.0 vs master).
methods_source = ast.parse(source_file.read_text(), filename=str(source_file))
test_class = next(x for x in methods_source.body if isinstance(x, ast.ClassDef) and x.name == "TestLearner")
targets = {
    "test_stale_paired_module_optimizer_does_not_load_optimizer",
    "test_equal_version_paired_state_cannot_only_restore_optimizer",
    "test_optimizer_only_future_version_does_not_publish_unseen_weights",
}
methods = [copy.deepcopy(x) for x in test_class.body
           if isinstance(x, ast.FunctionDef) and x.name in targets]
assert {x.name for x in methods} == targets, "Three native test methods not found"
for method in methods:
    method.decorator_list = []
    method.returns = None

source = ast.parse(learner_file.read_text(), filename=str(learner_file))
production = next(x for x in source.body if isinstance(x, ast.ClassDef) and x.name == "Learner")
set_state = copy.deepcopy(next(x for x in production.body if isinstance(x, ast.FunctionDef) and x.name == "set_state"))
set_state.decorator_list = []
set_state.returns = None
set_state.args.args[1].annotation = None
scope = dict(vars(learner_module))
exec(compile(ast.fix_missing_locations(ast.Module(body=[set_state], type_ignores=[])),
             str(learner_file), "exec"), scope)
learner_module.Learner.set_state = scope["set_state"]

ns = {
    "mock": mock,
    "np": np,
    "COMPONENT_OPTIMIZER": COMPONENT_OPTIMIZER,
    "COMPONENT_RL_MODULE": COMPONENT_RL_MODULE,
    "WEIGHTS_SEQ_NO": WEIGHTS_SEQ_NO,
    "BaseTestingAlgorithmConfig": BaseTestingAlgorithmConfig,
}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
             str(source_file), "exec"), ns)
Native = type("NativeTestLearner", (unittest.TestCase,),
              {"ENV": gym.make("CartPole-v1"), **{name: ns[name] for name in targets}})

print("Loaded 3 actual checked-out native TestLearner methods")
print("Ray wheel:", ray.__version__, ray.__file__)
print("Real Learner class:", learner_module.Learner.__module__)
print("Production method source:", learner_file)
print("Expected:", "three contract failures (baseline)" if baseline else "three passes (patch)")
ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
try:
    suite = unittest.TestLoader().loadTestsFromTestCase(Native)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = result.failures + result.errors
    if baseline:
        failed_names = {case._testMethodName for case, _ in failures}
        assert failed_names == targets and result.testsRun == 3, (
            result.testsRun, failed_names, targets
        )
        print("PASS red phase: all three official-source native tests reject unmodified behavior")
    else:
        assert result.wasSuccessful() and result.testsRun == 3, failures
        print("PASS green phase: all three official-source native tests pass the minimal patch")
finally:
    ray.shutdown()
    Native.ENV.close()
