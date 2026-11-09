"""No-crash negative control for warm Adam continuation after new DDP LearnerGroup.

This deliberately destroys a HEALTHY group; no Ray actor is killed. Compares
one-step continuation with a fresh group recreated from the same nonempty
optimizer checkpoint and the same batch. A completed diagnostic run is not
equivalent to a successful one-step trajectory parity result.
"""
from __future__ import annotations
import copy
import json
import time
import gymnasium as gym
import ray
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader
from rllib_native_two_learner_ddp_crash_health import (
    all_model_states, all_optimizer_states, optimizer_numeric_leaves,
    optimizer_max_abs_difference, maximum_error, all_optimizer_update_config,
)
def main():
    ray.init(num_cpus=4,include_dashboard=False,ignore_reinit_error=True,log_to_driver=False)
    env=gym.make("CartPole-v1")
    group=None
    try:
        config=BaseTestingAlgorithmConfig().learners(
            num_learners=2,num_cpus_per_learner=1,num_gpus_per_learner=0)
        group=config.build_learner_group(env=env)
        reader=get_cartpole_dataset_reader(batch_size=64)
        batch=reader.next().as_multi_agent()
        assert len(group.update(batch=batch,async_update=False))==2
        checkpoint=copy.deepcopy(group.get_state())
        warm=all_model_states(group)
        warm_opt=optimizer_numeric_leaves(checkpoint)
        warm_groups=all_optimizer_update_config(group)
        assert len(warm_opt)>=8
        assert maximum_error(*warm)<1e-6
        assert len(group.update(batch=batch,async_update=False))==2
        control=all_model_states(group)
        control_opt=optimizer_numeric_leaves(group.get_state())
        assert maximum_error(*control)<1e-6
        print("HEALTHY_REFERENCE_STEP",json.dumps({
            "grad_delta":maximum_error(warm[0],control[0]),
            "optimizer_buffers":len(warm_opt)}),flush=True)

        group.shutdown()
        group=None
        t0=time.perf_counter()
        group=config.build_learner_group(env=env)
        group.set_state(copy.deepcopy(checkpoint))
        fresh_ms=(time.perf_counter()-t0)*1000
        reconstructed=all_model_states(group)
        restored_opt=optimizer_numeric_leaves(group.get_state())
        restored_groups=all_optimizer_update_config(group)
        weight_err=maximum_error(warm[0],reconstructed[0])
        optimizer_err=optimizer_max_abs_difference(warm_opt,restored_opt)
        assert weight_err<1e-6,weight_err
        assert optimizer_err<1e-6,optimizer_err
        assert restored_groups==warm_groups
        print("NO_CRASH_NEW_GROUP_SNAPSHOT_RESTORED",json.dumps({
            "rebuild_and_restore_ms":fresh_ms,
            "weights_error":weight_err,
            "optimizer_error":optimizer_err,
            "param_groups_equal":True}),flush=True)

        assert len(group.update(batch=batch,async_update=False))==2
        next_models=all_model_states(group)
        next_opt=optimizer_numeric_leaves(group.get_state())
        next_err=maximum_error(control[0],next_models[0])
        peer_err=maximum_error(*next_models)
        opt_err=optimizer_max_abs_difference(control_opt,next_opt)
        print("NO_CRASH_FRESH_GROUP_CONTINUATION_RESULT",json.dumps({
            "next_model_ref_error":next_err,
            "next_optimizer_ref_error":opt_err,
            "new_group_peer_error":peer_err,
            "batch_rows":batch.count}),flush=True)
        if next_err>1e-5 or opt_err>1e-5:
            print("NO_CRASH_RECONSTRUCTION_TRAJECTORY_NON_EQUIVALENCE_OBSERVED",
                  flush=True)
        else:
            print("NO_CRASH_RECONSTRUCTION_TRAJECTORY_EQUIVALENCE_OBSERVED",
                  flush=True)
        print("NO_CRASH_CONTROL_COMPLETED",flush=True)
    finally:
        if group is not None:
            group.shutdown()
        env.close()
        ray.shutdown()
if __name__=="__main__":
    main()
