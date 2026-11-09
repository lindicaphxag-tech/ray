"""Native RLlib 2-Learner DDP causal optimizer-ownership ablation.

Hypothesis: native Ray checkpoint deserialization can expose NumPy buffers in
actor shared memory; CPU convert_to_torch_tensor uses torch.from_numpy, which
may retain storage ownership. An Adam optimizer mutates these buffers in place.

Compare unchanged standard restore with an actor-local deep-ownership copy of
ALL optimizer state before TorchLearner._set_optimizer_state.  The healthy
continued-training reference, original checkpoint, batch and DDP size are
identical in the two arms.

Diagnostic ONLY: no modifications to installed Ray source, no unproven claim
that read-only buffers are actually present or that ownership copying works.
"""
from __future__ import annotations

import copy
import json
import time

import gymnasium as gym
import numpy as np
import ray
import torch

from ray.rllib.core import COMPONENT_LEARNER
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader
from rllib_native_two_learner_ddp_crash_health import (
    all_model_states, maximum_error, optimizer_numeric_leaves,
    optimizer_max_abs_difference, all_optimizer_update_config
)


def actor_local_optimizer_ownership(learner):
    previous=learner._set_optimizer_state
    learner._ownership_log=[]
    def owned_restore(state):
        statistics={"numpy_leaf_count":0,"read_only_numpy_count":0,
                    "tensor_leaf_count":0,"copied_numpy_count":0,
                    "copied_tensor_count":0}
        def recursively_own(v):
            if isinstance(v,np.ndarray):
                statistics["numpy_leaf_count"]+=1
                if not v.flags.writeable:
                    statistics["read_only_numpy_count"]+=1
                copied=np.array(v,copy=True)
                assert copied.flags.writeable
                statistics["copied_numpy_count"]+=1
                return copied
            if torch.is_tensor(v):
                statistics["tensor_leaf_count"]+=1
                statistics["copied_tensor_count"]+=1
                return v.detach().clone()
            if isinstance(v,dict):
                return {k:recursively_own(item) for k,item in v.items()}
            if isinstance(v,list):
                return [recursively_own(item) for item in v]
            if isinstance(v,tuple):
                return tuple(recursively_own(item) for item in v)
            return copy.deepcopy(v)
        private_state=recursively_own(state)
        result=previous(private_state)
        learner._ownership_log.append(statistics)
        return result
    learner._set_optimizer_state=owned_restore
    return True


def get_actor_ownership_logs(learner):
    return list(getattr(learner,"_ownership_log",[]))


def snapshot_optimizer(group):
    return optimizer_numeric_leaves(group.get_state())


def main():
    ray.init(num_cpus=4,include_dashboard=False,
             ignore_reinit_error=True,log_to_driver=False)
    env=gym.make("CartPole-v1")
    group=None
    observations=[]
    try:
        cfg=BaseTestingAlgorithmConfig().learners(
            num_learners=2,num_cpus_per_learner=1,num_gpus_per_learner=0)
        group=cfg.build_learner_group(env=env)
        reader=get_cartpole_dataset_reader(batch_size=64)
        batch=reader.next().as_multi_agent()
        assert batch.count>0
        first=group.update(batch=batch,async_update=False)
        assert len(first)==2
        checkpoint=copy.deepcopy(group.get_state())
        warm_w=all_model_states(group)
        warm_o=snapshot_optimizer(group)
        warm_cfg=all_optimizer_update_config(group)
        assert len(warm_o)>=8
        assert maximum_error(*warm_w)<1e-5
        assert len(group.update(batch=batch,async_update=False))==2
        ref_w=all_model_states(group)
        ref_o=snapshot_optimizer(group)
        assert maximum_error(*ref_w)<1e-5
        print("OWNERSHIP_REFERENCE_STEP",json.dumps({
            "model_delta":maximum_error(warm_w[0],ref_w[0]),
            "buffers":len(warm_o),"rows":batch.count}),flush=True)
        group.shutdown()
        group=None

        for mode in ("stock_zero_copy","actor_local_owned_copy"):
            t0=time.perf_counter()
            group=cfg.build_learner_group(env=env)
            if mode=="actor_local_owned_copy":
                receipts=group.foreach_learner(actor_local_optimizer_ownership)
                assert len(receipts)==2 and all(r.ok and r.get() for r in receipts)
            group.set_state(copy.deepcopy(checkpoint))
            restored_w=all_model_states(group)
            restored_o=snapshot_optimizer(group)
            restored_cfg=all_optimizer_update_config(group)
            input_error=maximum_error(warm_w[0],restored_w[0])
            opt_input_error=optimizer_max_abs_difference(warm_o,restored_o)
            assert input_error<=1e-6,input_error
            assert opt_input_error<=1e-6,opt_input_error
            assert restored_cfg==warm_cfg
            ownership=([r.get() for r in group.foreach_learner(get_actor_ownership_logs)]
                       if mode=="actor_local_owned_copy" else [])
            elapsed=(time.perf_counter()-t0)*1000
            out=group.update(batch=batch,async_update=False)
            assert len(out)==2
            final_w=all_model_states(group)
            final_o=snapshot_optimizer(group)
            result={"mode":mode,
                "model_restore_error":input_error,
                "adam_restore_error":opt_input_error,
                "continuation_model_error":maximum_error(ref_w[0],final_w[0]),
                "continuation_adam_error":optimizer_max_abs_difference(ref_o,final_o),
                "within_ddp_model_mismatch":maximum_error(*final_w),
                "build_and_restore_ms":elapsed,
                "ownership_observations":ownership}
            observations.append(result)
            print("OWNERSHIP_ARM",json.dumps(result,sort_keys=True),flush=True)
            group.shutdown()
            group=None

        stock,owned=observations
        summary={"stock_model_error":stock["continuation_model_error"],
                 "owned_model_error":owned["continuation_model_error"],
                 "stock_ddp_peer_error":stock["within_ddp_model_mismatch"],
                 "owned_ddp_peer_error":owned["within_ddp_model_mismatch"],
                 "model_error_improvement":stock["continuation_model_error"]-
                                            owned["continuation_model_error"],
                 "optimizer_error_improvement":stock["continuation_adam_error"]-
                                                owned["continuation_adam_error"]}
        print("OPTIMIZER_BUFFER_OWNERSHIP_CAUSAL_ABLATION",json.dumps(summary),
              flush=True)
        print("OWNERSHIP_NATIVE_TWO_LEARNER_DIAGNOSTIC_COMPLETE",flush=True)
    finally:
        if group is not None:
            group.shutdown()
        env.close()
        ray.shutdown()


if __name__=="__main__":
    main()
