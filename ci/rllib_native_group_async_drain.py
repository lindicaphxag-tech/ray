"""Native RLlib LearnerGroup async-update quiescence contract.

This experiments on *actual* ray[rllib]==2.59.0 LearnerGroup.update(
async_update=True), its native FaultTolerantActorManager outstanding queue
and a real torch model. It does NOT monkeypatch upstream LearnerGroup,
and only a userland gate prevents NEW update submissions.

Invariant targeted: no externally submitted async update may survive a
checkpoint-install fence. Drain without calling LearnerGroup.update again,
because that API re-submits new work when async_update=True.
"""
from __future__ import annotations

import copy
import json
import pathlib
import time
import gymnasium as gym
import numpy as np
import ray

from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader


def install_deterministic_slow_gradient(learner):
    original=learner.before_gradient_based_update
    def delayed(*args, **kwargs):
        time.sleep(.75)
        return original(*args, **kwargs)
    learner.before_gradient_based_update=delayed
    return True


def model_floats(group):
    state=group.get_state()[COMPONENT_LEARNER][COMPONENT_RL_MODULE][DEFAULT_MODULE_ID]
    return {k:np.asarray(v).copy() for k,v in state.items()
            if np.issubdtype(np.asarray(v).dtype,np.floating)}


def max_change(a,b):
    return max(float(np.max(np.abs(b[k]-v))) for k,v in a.items())


class UserlandEpochFence:
    def __init__(self,group):
        self.group=group
        self.frozen=False
        self.epoch=1

    def update(self,batch):
        if self.frozen:
            raise RuntimeError("epoch fenced: training submission forbidden")
        return self.group.update(batch=batch,async_update=True)

    def drain_and_restore(self,authoritative,*,max_wait_s=25):
        self.frozen=True
        manager=self.group._worker_manager
        deadline=time.monotonic()+max_wait_s
        observed=manager.num_outstanding_async_reqs()
        drained=0
        while manager.num_outstanding_async_reqs()>0:
            if time.monotonic()>=deadline:
                raise TimeoutError("native async queue not quiescent")
            ready=manager.fetch_ready_async_reqs(timeout_seconds=.3)
            for res in ready:
                if not res.ok:
                    raise RuntimeError(f"Learner async update failed: {res.get()}")
                drained+=1
        assert manager.num_outstanding_async_reqs()==0
        # Only now is it safe to call real LearnerGroup.set_state from THIS
        # userland wrapper. Independent group callers could still race.
        self.group.set_state(copy.deepcopy(authoritative))
        self.epoch+=1
        self.frozen=False
        return {"observed":observed,"drained":drained,"published":self.epoch}


def main():
    ray.init(num_cpus=4,include_dashboard=False,
             ignore_reinit_error=True,log_to_driver=False)
    env=gym.make("CartPole-v1")
    group=None
    try:
        config=BaseTestingAlgorithmConfig().learners(
            num_learners=1,num_cpus_per_learner=1,num_gpus_per_learner=0)
        group=config.build_learner_group(env=env)
        assert group.is_remote and group._worker_manager is not None
        assertions=group.foreach_learner(install_deterministic_slow_gradient)
        # The real actor manager returns Ray ResultOrError elements.
        assert all(x.ok and x.get() for x in assertions)
        reader=get_cartpole_dataset_reader(batch_size=32)
        batch=reader.next().as_multi_agent()
        assert batch.count>0

        initial=group.get_state()
        m0=model_floats(group)
        fence=UserlandEpochFence(group)
        first=fence.update(batch)
        outstanding=group._worker_manager.num_outstanding_async_reqs()
        assert outstanding>=1, outstanding
        print("REAL_GROUP_ASYNC_SUBMITTED",
              json.dumps({"outstanding":outstanding,
                          "first_return_count":len(first),
                          "batch_rows":batch.count}),flush=True)

        # Freeze immediately. This is a wrapper admission check, not a
        # claim that arbitrary callers of the original group are blocked.
        fence.frozen=True
        try:
            fence.update(batch)
        except RuntimeError as err:
            assert "fenced" in str(err)
        else:
            raise AssertionError("admission failed to reject new async request")
        t0=time.perf_counter()
        result=fence.drain_and_restore(initial)
        elapsed=(time.perf_counter()-t0)*1000
        assert result["observed"]>=1 and result["drained"]>=1,result
        # The original state is restored, so policy tensor should match.
        m1=model_floats(group)
        assert max_change(m0,m1)<1e-5,max_change(m0,m1)
        print("REAL_GROUP_DRAIN_RESTORE",
              json.dumps({**result,"wall_ms":elapsed,
                          "restored_max_abs_error":max_change(m0,m1)}),
              flush=True)

        assert not fence.frozen
        second=fence.update(batch)
        assert group._worker_manager.num_outstanding_async_reqs()>=1
        # Drain second genuine native async gradient step without submitting
        # a third accidental update (a common unsafe drain pattern).
        fence.frozen=True
        deadline=time.monotonic()+30
        count=0
        while group._worker_manager.num_outstanding_async_reqs():
            assert time.monotonic()<deadline
            for r in group._worker_manager.fetch_ready_async_reqs(timeout_seconds=.3):
                assert r.ok,str(r.get())
                count+=1
        assert count>=1
        after=model_floats(group)
        delta=max_change(m1,after)
        assert delta>0,(delta,count)
        print("REAL_GROUP_ASYNC_TRAINED",
              json.dumps({"completed":count,"max_parameter_change":delta,
                          "batch_rows":batch.count}),flush=True)
        print("ACTUAL_RLLIB_LEARNERGROUP_ASYNC_QUIESCENCE_PASSED",flush=True)
    finally:
        if group is not None:
            group.shutdown()
        env.close()
        ray.shutdown()


if __name__=="__main__":
    main()
