"""Ray native-wheel smoke test for the exact fork Learner.set_state body.

The imported RLlib Learner, RLModule and runtime are a released Ray wheel,
while set_state is extracted verbatim from the fork's development source.
This is NOT a same-revision full upstream integration test.
"""
import ast
from pathlib import Path

import gymnasium as gym
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO


def patch_with_exact_development_method():
    path = Path("python/ray/rllib/core/learner/learner.py")
    tree = ast.parse(path.read_text())
    clazz = next(n for n in tree.body
                 if isinstance(n, ast.ClassDef) and n.name == "Learner")
    fn = next(n for n in clazz.body
              if isinstance(n, ast.FunctionDef) and n.name == "set_state")
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    exec_ns = {
        "COMPONENT_RL_MODULE": COMPONENT_RL_MODULE,
        "COMPONENT_OPTIMIZER": "optimizer",
        "COMPONENT_METRICS_LOGGER": "metrics_logger",
        "WEIGHTS_SEQ_NO": WEIGHTS_SEQ_NO,
    }
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), exec_ns)
    Learner.set_state = exec_ns["set_state"]


def main():
    import ray
    print("RAY_RELEASE_RUNTIME", ray.__version__, flush=True)
    patch_with_exact_development_method()
    env = gym.make("CartPole-v1")
    learner = BaseTestingAlgorithmConfig().build_learner(env=env)
    original = learner.get_state()
    assert original[WEIGHTS_SEQ_NO] == 0, original[WEIGHTS_SEQ_NO]
    module = original[COMPONENT_RL_MODULE]
    learner.set_state({COMPONENT_RL_MODULE: module, WEIGHTS_SEQ_NO: 0})
    assert learner.get_state()[WEIGHTS_SEQ_NO] == 0
    learner.set_state({COMPONENT_RL_MODULE: module})
    assert learner.get_state()[WEIGHTS_SEQ_NO] == 1
    learner.set_state({COMPONENT_RL_MODULE: module, WEIGHTS_SEQ_NO: 0})
    assert learner.get_state()[WEIGHTS_SEQ_NO] == 2
    print("REAL_LEARNER_METHOD_ROUND_TRIP_GREEN", flush=True)
    env.close()


if __name__ == "__main__":
    main()
