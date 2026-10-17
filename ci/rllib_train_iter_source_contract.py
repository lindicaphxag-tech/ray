"""Dependency-free execution of the actual RLlib TrainIterCtx from the checked-out source.

This is a narrow production-method smoke check, not the distributed RLlib suite.
"""
import ast
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

algorithm_path = Path("python/ray/rllib/algorithms/algorithm.py")
config_path = Path("python/ray/rllib/algorithms/algorithm_config.py")
algorithm_ast = ast.parse(algorithm_path.read_text(encoding="utf-8"), filename=str(algorithm_path))
config_ast = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
class_node = next(n for n in algorithm_ast.body if isinstance(n, ast.ClassDef) and n.name == "TrainIterCtx")
module = ast.fix_missing_locations(ast.Module(body=[class_node], type_ignores=[]))
constants = (
    "ALL_MODULES", "ENV_RUNNER_RESULTS", "LEARNER_RESULTS",
    "NUM_AGENT_STEPS_SAMPLED", "NUM_AGENT_STEPS_SAMPLED_LIFETIME",
    "NUM_AGENT_STEPS_TRAINED", "NUM_AGENT_STEPS_TRAINED_LIFETIME",
    "NUM_ENV_STEPS_SAMPLED", "NUM_ENV_STEPS_SAMPLED_LIFETIME",
    "NUM_ENV_STEPS_TRAINED", "NUM_ENV_STEPS_TRAINED_LIFETIME",
)
scope = {"Algorithm": object, "np": math, "time": time}
scope.update({key: key for key in constants})
exec(compile(module, str(algorithm_path), "exec"), scope)
TrainIterCtx = scope["TrainIterCtx"]

config_node = next(n for n in config_ast.body if isinstance(n, ast.ClassDef) and n.name == "AlgorithmConfig")
initializer = next(n for n in config_node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
fault_tolerance = next(n for n in config_node.body if isinstance(n, ast.FunctionDef) and n.name == "fault_tolerance")
budget_name = "num_consecutive_sample_no_progress_tolerance"
assert any(arg.arg == budget_name for arg in fault_tolerance.args.kwonlyargs), "Independent budget must be configurable"
assert any(
    isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == budget_name for t in node.targets
    )
    for node in ast.walk(initializer)
), "Missing configured default budget"

ER = scope["ENV_RUNNER_RESULTS"]
LR = scope["LEARNER_RESULTS"]
AL = scope["ALL_MODULES"]
ES = scope["NUM_ENV_STEPS_SAMPLED"]
ET = scope["NUM_ENV_STEPS_TRAINED"]
AS = scope["NUM_AGENT_STEPS_SAMPLED"]
AT = scope["NUM_AGENT_STEPS_TRAINED"]
ESL = scope["NUM_ENV_STEPS_SAMPLED_LIFETIME"]
ETL = scope["NUM_ENV_STEPS_TRAINED_LIFETIME"]
ASL = scope["NUM_AGENT_STEPS_SAMPLED_LIFETIME"]
ATL = scope["NUM_AGENT_STEPS_TRAINED_LIFETIME"]

class Metrics:
    def __init__(self):
        self.values = {
            (ER, ESL): 0, (LR, AL, ETL): 0,
            (ER, ASL): {}, (LR, ATL): {},
        }
    def peek(self, key, default=None):
        return self.values.get(key, default)

def fake_algo(*, budget=1, worker_failures=100, timeout=1.0,
              min_sample=3, min_train=0, new_api=False, step_mode="env_steps"):
    config = SimpleNamespace(
        enable_env_runner_and_connector_v2=new_api,
        count_steps_by=step_mode,
        min_time_s_per_iteration=None,
        min_sample_timesteps_per_iteration=min_sample,
        min_train_timesteps_per_iteration=min_train,
        num_consecutive_env_runner_failures_tolerance=worker_failures,
        num_consecutive_sample_no_progress_tolerance=budget,
        sample_timeout_s=timeout,
    )
    return SimpleNamespace(config=config, _counters=defaultdict(int), metrics=Metrics())

def raises(exception, fn, message):
    try:
        fn()
    except exception as exc:
        assert message in str(exc), str(exc)
        return
    raise AssertionError(f"Expected {exception.__name__}: {message}")

algo = fake_algo(budget=1, worker_failures=100)
with TrainIterCtx(algo) as ctx:
    assert ctx.should_stop(None) is False
    assert ctx.should_stop(True) is False
    raises(RuntimeError, lambda: ctx.should_stop(True), "No sampling progress")
print("PASS: sampling-stall budget is independent of worker retry budget")

algo = fake_algo(budget=100, worker_failures=0)
with TrainIterCtx(algo) as ctx:
    assert ctx.should_stop(None) is False
    raises(RuntimeError, lambda: ctx.should_stop(False), "consecutive worker failures")
print("PASS: worker failure tolerance retains its original behavior")

algo = fake_algo(budget=1)
with TrainIterCtx(algo) as ctx:
    assert ctx.should_stop(None) is False
    assert ctx.should_stop(True) is False
    algo._counters[ES] = 1
    assert ctx.should_stop(True) is False
    assert ctx.sample_progress_failures == 0
    assert ctx.should_stop(True) is False
    raises(RuntimeError, lambda: ctx.should_stop(True), "No sampling progress")
print("PASS: progress resets no-progress counter")

algo = fake_algo(budget=0, timeout=None)
with TrainIterCtx(algo) as ctx:
    assert ctx.should_stop(None) is False
    for _ in range(7):
        assert ctx.should_stop(True) is False
    assert ctx.sample_progress_failures == 0
print("PASS: blocking sampling is not spuriously rejected")

algo = fake_algo(budget=0, min_sample=3, min_train=100)
with TrainIterCtx(algo) as ctx:
    assert ctx.should_stop(None) is False
    algo._counters[ES] = 3
    for _ in range(7):
        assert ctx.should_stop(True) is False
    assert ctx.sample_progress_failures == 0
print("PASS: watchdog disarms at sampling target even while waiting for training target")

algo = fake_algo(budget=1, new_api=True, step_mode="agent_steps")
algo.metrics.values[(ER, ASL)] = {"active": 3, "retired": float("nan")}
algo.metrics.values[(LR, ATL)] = {"active": 5, "retired": float("nan")}
with TrainIterCtx(algo) as ctx:
    assert ctx.should_stop(None) is False
    assert ctx.should_stop(True) is False
    algo.metrics.values[(ER, ASL)] = {"active": 4, "retired": float("nan")}
    algo.metrics.values[(LR, ATL)] = {"active": 6, "retired": float("nan")}
    assert ctx.should_stop(True) is False
    assert (ctx.sampled, ctx.trained) == (1, 1)
    assert ctx.sample_progress_failures == 0
print("PASS: new-stack agent metrics ignore missing retired-agent NaNs")

algo = fake_algo(new_api=True, step_mode="agent_steps")
algo.metrics.values[(ER, ASL)] = {"active": 3, "invalid": float("inf")}
raises(ValueError, lambda: TrainIterCtx(algo).__enter__(), "Infinite agent-step")
print("PASS: invalid infinite lifetime counters fail closed")

print("PASS: production source AST compiled and all 7 RLlib contracts checked")
