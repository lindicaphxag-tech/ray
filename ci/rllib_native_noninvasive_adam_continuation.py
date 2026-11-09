"""Noninvasive checkpoint continuation test on actual Ray 2.59 CPU Learner.

NO monkeypatch of gradients, optimizer, LR schedulers or Learner internals.
Pair a live warmed native Learner with a fresh native Learner restored from
its full checkpoint. Both process the same immutable CartPole sample with
identical CPU RNG snapshots; actual updates are unmodified RLlib production.
Print all observed errors rather than asserting the desired equality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random

import gymnasium as gym
import numpy as np
import ray
import torch
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader


def flatten_numeric(value, prefix=()):
    out={}
    if isinstance(value,dict):
        for k,v in value.items():
            out.update(flatten_numeric(v,prefix+(str(k),)))
    elif isinstance(value,(list,tuple)):
        for k,v in enumerate(value):
            out.update(flatten_numeric(v,prefix+(str(k),)))
    else:
        if hasattr(value,"detach"):
            value=value.detach().cpu().numpy()
        try:
            x=np.asarray(value)
            if x.dtype.kind in "biufc" and x.size:
                out["/".join(prefix)]=np.array(x,copy=True)
        except (TypeError,ValueError):
            pass
    return out


def diff(a,b):
    assert a.keys()==b.keys(),(list(a.keys()-b.keys())[:5],list(b.keys()-a.keys())[:5])
    maxi=0.0;changed=0
    for k in a:
        x=np.asarray(a[k]);y=np.asarray(b[k])
        assert x.shape==y.shape,(k,x.shape,y.shape)
        if x.dtype.kind in "biufc":
            d=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))
            if math.isnan(d): raise AssertionError("NaN difference at "+k)
            maxi=max(maxi,d)
            changed+=d!=0
    return {"max_abs":maxi,"changed_leaves":changed,"total_leaves":len(a)}


def state_parts(state):
    return (flatten_numeric(state[COMPONENT_RL_MODULE]),
            flatten_numeric(state[COMPONENT_OPTIMIZER]))


def capture_rng():
    return (torch.get_rng_state().clone(),np.random.get_state(),random.getstate())


def restore_rng(rng):
    torch.set_rng_state(rng[0].clone())
    np.random.set_state(rng[1])
    random.setstate(rng[2])


def live_adam(learner):
    names={id(p):n for n,p in learner.module[DEFAULT_MODULE_ID].named_parameters()}
    moments={}
    for o_name,opt in sorted(learner._named_optimizers.items()):
        for p,st in opt.state.items():
            n=names.get(id(p),"<NOT_A_LIVE_MODEL_PARAMETER>")
            for attr,v in st.items():
                moments[o_name+"/"+n+"/"+str(attr)]=v
    assert moments, "warm checkpoint did not create optimizer state"
    return flatten_numeric(moments)


def one_trial(seed,raw_batch):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    env=gym.make("CartPole-v1")
    try:
        config=BaseTestingAlgorithmConfig()
        reference=config.build_learner(env=env)
        batch0=reference._convert_batch_type(raw_batch)
        assert batch0.count>0
        reference.update(batch=batch0)
        saved=copy.deepcopy(reference.get_state())
        s0,o0=state_parts(saved)
        l0=live_adam(reference)
        candidate=config.build_learner(env=env)
        candidate.set_state(copy.deepcopy(saved))
        before=candidate.get_state()
        s1,o1=state_parts(before)
        l1=live_adam(candidate)
        entry={"seed":seed,"batch_rows":batch0.count,
               "model_pre":diff(s0,s1),
               "optimizer_pre":diff(o0,o1),
               "live_optimizer_pre":diff(l0,l1)}
        # Same source arrays, immutable batch per learner, isolated RNG.
        reference_batch=reference._convert_batch_type(copy.deepcopy(raw_batch))
        candidate_batch=candidate._convert_batch_type(copy.deepcopy(raw_batch))
        rng=capture_rng()
        restore_rng(rng)
        reference.update(batch=reference_batch)
        restore_rng(rng)
        candidate.update(batch=candidate_batch)
        sr,orr=state_parts(reference.get_state())
        sc,orc=state_parts(candidate.get_state())
        entry.update({"model_next":diff(sr,sc),
                      "optimizer_next":diff(orr,orc),
                      "live_optimizer_next":diff(live_adam(reference),live_adam(candidate))})
        return entry
    finally:
        env.close()


def main():
    torch.set_num_threads(1)
    ray.init(num_cpus=2,include_dashboard=False,ignore_reinit_error=True,
             log_to_driver=False)
    try:
        batch=get_cartpole_dataset_reader(batch_size=32).next().as_multi_agent()
        assert batch.count>0
        for seed in [7,19,31,43]:
            report=one_trial(seed,batch)
            print("NONINVASIVE_NATIVE_LEARNER_TRIAL",json.dumps(report,sort_keys=True),flush=True)
        print("NONINVASIVE_NATIVE_LEARNER_EXPERIMENT_COMPLETE",flush=True)
    finally:
        ray.shutdown()

if __name__=="__main__":
    main()
