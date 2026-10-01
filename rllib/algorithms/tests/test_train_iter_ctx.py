from types import SimpleNamespace

import pytest

from ray.rllib.algorithms.algorithm import TrainIterCtx
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_TRAINED,
)


def _make_train_iter_ctx(no_sample_steps_tolerance=1):
    algo = SimpleNamespace(
        config=SimpleNamespace(
            enable_env_runner_and_connector_v2=False,
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
    ctx = TrainIterCtx(algo)
    ctx.__enter__()
    return algo, ctx


def test_train_iteration_raises_after_consecutive_steps_without_samples():
    _, ctx = _make_train_iter_ctx(no_sample_steps_tolerance=1)

    assert not ctx.should_stop(False)
    assert not ctx.should_stop(True)
    with pytest.raises(RuntimeError, match="has not sampled any new timesteps"):
        ctx.should_stop(True)


def test_train_iteration_resets_no_sample_count_when_sampling_progresses():
    algo, ctx = _make_train_iter_ctx(no_sample_steps_tolerance=1)

    assert not ctx.should_stop(False)
    assert not ctx.should_stop(True)
    algo._counters[NUM_ENV_STEPS_SAMPLED] = 1
    assert not ctx.should_stop(True)
    assert not ctx.should_stop(True)
    algo._counters[NUM_ENV_STEPS_SAMPLED] = 2
    assert ctx.should_stop(True)
