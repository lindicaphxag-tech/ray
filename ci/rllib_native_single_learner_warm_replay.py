"""Controlled decomposition of RLlib warm checkpoint transition equivalence.

Run two genuinely different control paths in one native CPU job:
  1. Pure PyTorch Adam with identical tensors/gradients/loaded state (positive).
  2. A *single* native Ray RLlib Learner, uninterrupted continuation vs a
     fresh Learner restored from the same warm checkpoint (diagnostic).

No DDP, no physical actor failure, no optimizer.step monkeypatching.
This script records a negative single-Learner observation without pretending
a numeric trajectory test passed. Positive pure Torch replay MUST pass.
"""
from __future__ import annotations

import copy
import json

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader

from rllib_native_two_learner_ddp_crash_health import (
    optimizer_max_abs_difference,optimizer_numeric_leaves,
)


def pure_torch_positive_control():
    torch.manual_seed(29341)
    p=[torch.nn.Parameter(torch.randn(*size))
       for size in ((32,4),(32,),(2,32),(2,))]
    opt=torch.optim.Adam(p,lr=0.001,betas=(0.9,0.999),eps=1e-8)
    for _ in range(3):
        for par in p:
            par.grad=torch.randn_like(par)*0.03
        opt.step()
    snapshot=copy.deepcopy(opt.state_dict())
    q=[torch.nn.Parameter(par.detach().clone()) for par in p]
    other=torch.optim.Adam(q,lr=0.001,betas=(0.9,0.999),eps=1e-8)
    other.load_state_dict(snapshot)
    gradients=[torch.randn_like(par)*0.07 for par in p]
    for pa,pb,grad in zip(p,q,gradients):
        pa.grad=grad.clone()
        pb.grad=grad.clone()
    opt.step()
    other.step()
    delta=max(float((pa-pb).abs().max().item()) for pa,pb in zip(p,q))
    assert delta==0.0,delta
    moment_delta=0.0
    for idx in range(len(p)):
        for field in ("step","exp_avg","exp_avg_sq"):
            moment_delta=max(moment_delta,float((
                opt.state[p[idx]][field]-other.state[q[idx]][field]
            ).abs().max().item()))
    assert moment_delta==0.0,moment_delta
    print("PURE_TORCH_ADAM_POSITIVE",
          json.dumps({"next_param_max_abs_error":delta,
                      "next_state_max_abs_error":moment_delta,
                      "parameters":len(p),"warmup_steps":3}),flush=True)


def model_arrays(learner):
    state=learner.get_state()[COMPONENT_RL_MODULE][DEFAULT_MODULE_ID]
    return {name:np.asarray(value).copy() for name,value in state.items()
            if np.issubdtype(np.asarray(value).dtype,np.floating)}


def error(x,y):
    assert x.keys()==y.keys(),(x.keys(),y.keys())
    return max(float(np.max(np.abs(x[k]-y[k]))) for k in x)


def optimizer_arrays(learner):
    return optimizer_numeric_leaves({COMPONENT_LEARNER:learner.get_state()})


def update(learner,batch):
    # Prevent update in either Learner from modifying the shared input object.
    learner.update(batch=learner._convert_batch_type(copy.deepcopy(batch)))


def main():
    pure_torch_positive_control()
    env=gym.make("CartPole-v1")
    try:
        config=BaseTestingAlgorithmConfig()
        learner=config.build_learner(env=env)
        reader=get_cartpole_dataset_reader(batch_size=64)
        batch=reader.next().as_multi_agent()
        assert batch.count>=1
        update(learner,batch)
        warm=copy.deepcopy(learner.get_state())
        weights=model_arrays(learner)
        optim=optimizer_arrays(learner)
        assert len(optim)>=8,len(optim)
        update(learner,batch)
        reference_w=model_arrays(learner)
        reference_o=optimizer_arrays(learner)
        reference_delta=error(weights,reference_w)
        assert reference_delta>0,reference_delta

        new_learner=config.build_learner(env=env)
        new_learner.set_state(copy.deepcopy(warm))
        restored_w=model_arrays(new_learner)
        restored_o=optimizer_arrays(new_learner)
        weight_restore_error=error(weights,restored_w)
        opt_restore_error=optimizer_max_abs_difference(optim,restored_o)
        assert weight_restore_error<=1e-6,weight_restore_error
        assert opt_restore_error<=1e-6,opt_restore_error

        update(new_learner,batch)
        after_w=model_arrays(new_learner)
        after_o=optimizer_arrays(new_learner)
        step_error=error(reference_w,after_w)
        adam_error=optimizer_max_abs_difference(reference_o,after_o)
        print("SINGLE_RLLIB_NATIVE_WARM_RESTORE",
              json.dumps({"restored_model_error":weight_restore_error,
                          "restored_adam_error":opt_restore_error,
                          "next_model_error":step_error,
                          "next_adam_error":adam_error,
                          "reference_update_magnitude":reference_delta,
                          "optimizer_buffers":len(optim),
                          "batch_rows":batch.count}),flush=True)
        if step_error>1e-5 or adam_error>1e-5:
            print("SINGLE_LEARNER_TRANSITION_NON_EQUIVALENCE_OBSERVED",flush=True)
        else:
            print("SINGLE_LEARNER_TRANSITION_EQUIVALENCE_OBSERVED",flush=True)
        print("ONE_LEARNER_DIAGNOSTIC_COMPLETE",flush=True)
    finally:
        env.close()


if __name__=="__main__":
    main()
