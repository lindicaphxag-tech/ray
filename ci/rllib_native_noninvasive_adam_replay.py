"""Non-invasive Adam transition replay on real 2-Learner Ray DDP.

Imports the established physical-actor-kill/continuation test, but replaces its
optimizer.step monkeypatch with an observer attached only to the enclosing
Learner.apply_gradients entry.  During the NATIVE step, the original bound
TorchLearner.apply_gradients and Adam.step execute with NO monkeypatches.
A cloned Adam starts its numerical replay only AFTER the native step returns.

This separates an instrumentation observer effect from incomplete optimizer
checkpoint state. It does not fix any production bug.
"""
from __future__ import annotations

import copy
import hashlib
import json
import torch

import rllib_native_two_learner_ddp_crash_health as witness

original_install=witness.install_batch_provenance


def noninvasive_install(learner):
    """Install batch/gradient tracing, replace invasive step hook, not Adam."""
    original_install(learner)
    native_apply=type(learner).apply_gradients.__get__(learner,type(learner))
    learner._audit_adam_apply=[]

    def at_apply_boundary(gradients_dict):
        observations=[]
        for optimizer_name,optim in sorted(learner._named_optimizers.items()):
            live_params=[p for grp in optim.param_groups for p in grp["params"]]
            names={id(p):name for name,p in
                   learner.module[witness.DEFAULT_MODULE_ID].named_parameters()}
            grouped=[]
            offset=0
            fresh_params=[]
            native_before={}
            input_grads=[]
            for group in optim.param_groups:
                targets=[]
                for original_param in group["params"]:
                    label=names.get(id(original_param),"<UNBOUND>")
                    assert label!="<UNBOUND>",label
                    clone=torch.nn.Parameter(original_param.detach().clone())
                    targets.append(clone)
                    fresh_params.append(clone)
                    native_before[label]=hashlib.sha256(
                        original_param.detach().cpu().numpy().tobytes()).hexdigest()
                    g=gradients_dict.get(original_param)
                    assert g is not None and bool(torch.isfinite(g).all()),label
                    input_grads.append(g.detach().clone())
                groups_hparams={
                    name:copy.deepcopy(value)
                    for name,value in group.items() if name!="params"}
                groups_hparams["params"]=targets
                grouped.append(groups_hparams)
            assert len(fresh_params)==len(live_params)
            # Deep-copy exact numerical optimizer checkpoint *before* native
            # update, and do not create or step the clone until native returns.
            state_saved=copy.deepcopy(optim.state_dict())
            observations.append((optimizer_name,optim,names,live_params,fresh_params,
                                 input_grads,grouped,state_saved,native_before))

        # This is the actual stock TorchLearner.update gradient application,
        # with the ACTUAL optimizer.step untouched and no extra torch.step.
        native_result=native_apply(gradients_dict)

        for (optimizer_name,optim,names,live_params,targets,input_grads,
             grouped,state_saved,native_before) in observations:
            native_after={
                names[id(p)]:p.detach().clone() for p in live_params}
            independent=type(optim)(grouped)
            independent.load_state_dict(state_saved)
            for target,grad in zip(targets,input_grads):
                target.grad=grad
            independent.step()
            diffs={}
            for p,q in zip(live_params,targets):
                label=names[id(p)]
                diffs[label]=float(torch.max(
                    torch.abs(native_after[label]-q.detach())).item())
            learner._audit_adam_apply.append({
                "optimizer":optimizer_name,
                "noninvasive_adam_replay_max_abs_error":max(diffs.values()),
                "per_parameter_errors":diffs,
                "native_before":native_before,
                "native_after":{key:hashlib.sha256(
                    p.detach().cpu().numpy().tobytes()).hexdigest()
                    for key,p in native_after.items()},
                "actual_gradient_sha256":{
                    names[id(p)]:hashlib.sha256(g.detach().cpu().numpy().tobytes()).hexdigest()
                    for p,g in zip(live_params,input_grads)},
            })
        return native_result

    learner.apply_gradients=at_apply_boundary
    return True


def main():
    witness.install_batch_provenance=noninvasive_install
    witness.main()


if __name__=="__main__":
    main()
