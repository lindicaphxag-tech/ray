from types import SimpleNamespace

import pytest

from ray.rllib.algorithms.algorithm import TrainIterCtx
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_TRAINED,
)


class _FakeMetrics:
    def __init__(self):
        self.values = {}

    def peek(self, key, default=None):
        return self.values.get(key, default)


def _make_train_iter_ctx(no_sample_steps_tolerance=1, new_api=False):
    algo = SimpleNamespace(
        config=SimpleNamespace(
            enable_env_runner_and_connector_v2=new_api,
            count_steps_by="env_steps",
            min_time_s_per_iteration=0,
            min_sample_timesteps_per_iteration=2,
            min_train_timesteps_per_iteration=0,
            num_consecutive_env_runner_failures_tolerance=100,
            num_consecutive_no_sample_steps_tolerance=no_sample_steps_tolerance,
        ),
        _counters={
            NUM_AGENT_STEPS_SAMPLED: 0,
            NUM_AGENT_STEPS_TRAINED: 0,
            NUM_ENV_STEPS_SAMPLED: 0,
            NUM_ENV_STEPS_TRAINED: 0,
        },
    )
    if new_api:
        algo.metrics = _FakeMetrics()
    ctx = TrainIterCtx(algo)
    ctx.__enter__()
    return algo, ctx


def _set_sampled(algo, timesteps):
    if hasattr(algo, "metrics"):
        algo.metrics.values[
            (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME)
        ] = timesteps
    else:
        algo._counters[NUM_ENV_STEPS_SAMPLED] = timesteps


@pytest.mark.parametrize("new_api", [False, True])
def test_train_iteration_raises_after_consecutive_steps_without_samples(new_api):
    _, ctx = _make_train_iter_ctx(no_sample_steps_tolerance=1, new_api=new_api)

    assert not ctx.should_stop(False)
    assert not ctx.should_stop(True)
    with pytest.raises(RuntimeError, match="has not sampled any new timesteps"):
        ctx.should_stop(True)


@pytest.mark.parametrize("new_api", [False, True])
def test_train_iteration_resets_no_sample_count_when_sampling_progresses(new_api):
    algo, ctx = _make_train_iter_ctx(no_sample_steps_tolerance=1, new_api=new_api)

    assert not ctx.should_stop(False)
    assert not ctx.should_stop(True)
    _set_sampled(algo, 1)
    assert not ctx.should_stop(True)
    assert not ctx.should_stop(True)
    _set_sampled(algo, 2)
    assert ctx.should_stop(True)
