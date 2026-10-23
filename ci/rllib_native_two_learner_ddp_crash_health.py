"""Native two-Learner RLlib asynchronous training / stale actor-health witness.

Native Ray 2.59, torch CPU DDP; no production modifications. Demonstrates that
the async queue and health metadata are not a fresh proof of actor liveness.
The loss probe is after one successful distributed update and restore.
"""
from __future__ import annotations

import copy
import hashlib
import json
import secrets
import time

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_RL_MODULE, COMPONENT_OPTIMIZER, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader


def install_batch_provenance(learner):
    """Record the *actual per-Learner objective batch*, not only input batch."""
    original=learner.compute_loss_for_module
    learner._audit_batch_digest=[]
    def traced(*,module_id,config,batch,fwd_out):
        h=hashlib.sha256()
        names=[]
        for key in sorted(batch):
            value=batch[key]
            if hasattr(value,"detach"):
                value=value.detach().cpu().numpy()
            arr=np.asarray(value)
            if arr.dtype.kind not in "biufc":
                continue
            h.update(str(key).encode())
            h.update(str(arr.shape).encode())
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
            names.append(str(key))
        learner._audit_batch_digest.append(
            {"module":str(module_id),"sha256":h.hexdigest(),
             "field_count":len(names)})
        return original(module_id=module_id,config=config,batch=batch,fwd_out=fwd_out)
    learner.compute_loss_for_module=traced
    return True


def get_batch_provenance(learner):
    return list(getattr(learner,"_audit_batch_digest",[]))


def traced_batches(group):
    result=group.foreach_learner(get_batch_provenance)
    return [r.get() for r in result]


def delayed_gradient(learner):
    previous = learner.before_gradient_based_update
    def injected(*args, **kwargs):
        time.sleep(.45)
        return previous(*args, **kwargs)
    learner.before_gradient_based_update = injected
    return True


def model_state(learner):
    d=learner.get_state()[COMPONENT_RL_MODULE][DEFAULT_MODULE_ID]
    return {key:np.asarray(val).copy() for key,val in d.items()
            if np.issubdtype(np.asarray(val).dtype,np.floating)}


def maximum_error(a,b):
    assert a.keys()==b.keys()
    return max(float(np.max(np.abs(v-b[k]))) for k,v in a.items())


def all_model_states(group):
    return [r.get() for r in group.foreach_learner(model_state)]


def optimizer_numeric_leaves(snapshot):
    """Numeric optimizer buffers from one real Learner; no metrics or weights."""
    state=snapshot[COMPONENT_LEARNER][COMPONENT_OPTIMIZER]
    leaves={}
    def walk(value,path):
        if isinstance(value,dict):
            for k,v in value.items():
                walk(v,path+(str(k),))
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value):
                walk(v,path+(str(i),))
        else:
            arr=np.asarray(value)
            if arr.size and arr.dtype.kind in "biufc":
                leaves["/".join(path)]=arr.copy()
    for name,optimizer in state.items():
        # Explicitly include *only* optimizer moment/step dictionaries:
        # param_groups scalar hyperparameters are not proof of warm state.
        buffers=optimizer["state"]["state"]
        assert buffers,("empty Adam moments",name)
        walk(buffers,(str(name),"state"))
    assert leaves,"no numeric optimizer buffer state found"
    return leaves


def optimizer_max_abs_difference(a,b):
    assert a.keys()==b.keys(),(a.keys()-b.keys(),b.keys()-a.keys())
    return max(float(np.max(np.abs(
        np.asarray(a[k],dtype=np.float64)-np.asarray(b[k],dtype=np.float64))))
        for k in a)


def all_optimizer_states(group):
    return [r.get() for r in group.foreach_learner(
        lambda learner: optimizer_numeric_leaves({
            COMPONENT_LEARNER:learner.get_state()
        }))]


def checked_optimizer_error(a,b,*,atol=1e-6,rtol=1e-5):
    assert a.keys()==b.keys(),(a.keys()-b.keys(),b.keys()-a.keys())
    max_delta=0.0
    for k in a:
        av,bv=np.asarray(a[k]),np.asarray(b[k])
        np.testing.assert_allclose(bv,av,atol=atol,rtol=rtol,err_msg=k)
        max_delta=max(max_delta,float(np.max(np.abs(
            av.astype(np.float64)-bv.astype(np.float64)))))
    return max_delta


def fresh_membership_attestation(group,expected_ids):
    """Return a per-call membership ACK from actual managed Ray actors.

    This is a liveness challenge at a point in time, NOT a durable lease,
    cryptographic identity proof or full commit barrier.
    """
    manager=group._worker_manager
    assert manager.num_outstanding_async_reqs()==0
    nonce=secrets.token_hex(12)
    call=group.foreach_learner(
        lambda learner, challenge=nonce: {
            "nonce":challenge,
            "module_ids":sorted(learner.module.keys()),
        },
        healthy_only=False,timeout_seconds=10)
    responses=list(call)
    response_ids={r.actor_id for r in responses}
    bad=[str(r.get())[:180] for r in responses if not r.ok]
    if (len(responses)!=len(expected_ids) or response_ids!=set(expected_ids)
            or bad or any(r.get().get("nonce")!=nonce for r in responses if r.ok)):
        raise RuntimeError(
            "fresh membership ACK failed: ids="
            +repr(sorted(response_ids))+" errors="+repr(bad))
    return {"ids":sorted(response_ids),"acks":len(responses),
            "challenge":nonce,"module_ids":responses[0].get()["module_ids"]}


class ExclusiveUserlandGate:
    def __init__(self,group):
        self.group=group
        self.closed=False

    def submit(self,batch):
        if self.closed:
            raise RuntimeError("group checkpoint fence has closed admissions")
        return self.group.update(batch=batch,async_update=True)

    def drain(self,timeout=40):
        self.closed=True
        m=self.group._worker_manager
        deadline=time.monotonic()+timeout
        observed=m.num_outstanding_async_reqs()
        done=0
        while m.num_outstanding_async_reqs():
            assert time.monotonic()<deadline,"DDP updates did not quiesce"
            ready=m.fetch_ready_async_reqs(timeout_seconds=.5)
            for r in ready:
                if not r.ok:
                    raise RuntimeError(f"async distributed Learner error: {r.get()}")
                done+=1
        assert m.num_outstanding_async_reqs()==0
        assert m.num_healthy_actors()==m.num_actors(),"unhealthy DDP group"
        return {"observed":observed,"completed":done,
                "count":m.num_actors(),"healthy":m.num_healthy_actors()}


def main():
    ray.init(num_cpus=4,include_dashboard=False,ignore_reinit_error=True,
             log_to_driver=False)
    env=gym.make("CartPole-v1")
    group=None
    killed=False
    try:
        config=BaseTestingAlgorithmConfig().learners(
            num_learners=2,num_cpus_per_learner=1,num_gpus_per_learner=0)
        group=config.build_learner_group(env=env)
        manager=group._worker_manager
        assert group.is_remote and manager.num_actors()==2
        res=group.foreach_learner(delayed_gradient)
        assert len(res)==2 and all(x.ok and x.get() for x in res),res
        baseline=group.get_state()
        models0=all_model_states(group)
        assert len(models0)==2
        assert maximum_error(*models0)<1e-6
        reader=get_cartpole_dataset_reader(batch_size=64)
        batch=reader.next().as_multi_agent()
        gate=ExclusiveUserlandGate(group)
        initial=gate.submit(batch)
        waiting=manager.num_outstanding_async_reqs()
        assert waiting==2,waiting
        assert len(initial)==0,initial
        gate.closed=True
        try:
            gate.submit(batch)
        except RuntimeError as exc:
            assert "closed" in str(exc)
        else:
            raise AssertionError("fenced submission was not rejected")
        drained=gate.drain()
        assert drained["completed"]==2,drained
        models1=all_model_states(group)
        assert maximum_error(*models1)<1e-5,maximum_error(*models1)
        delta=maximum_error(models0[0],models1[0])
        assert delta>0,delta
        print("TWO_DDP_LEARNERS_UPDATED",json.dumps(
            {"queue":drained,"batch_rows":batch.count,
             "max_grad_update":delta,"ddp_max_weight_mismatch":maximum_error(*models1)}),
            flush=True)
        # Real LearnerGroup remote checkpoint restore after all 2 DDP jobs finish.
        ordinary_start=time.perf_counter()
        group.set_state(copy.deepcopy(baseline))
        ordinary_restore_ms=(time.perf_counter()-ordinary_start)*1000
        models2=all_model_states(group)
        assert maximum_error(*models2)<1e-6
        restore_error=maximum_error(models0[0],models2[0])
        assert restore_error<1e-5,restore_error
        print("TWO_DDP_LEARNERS_RESTORED",
              json.dumps({"restored_max_error":restore_error,
                          "peer_mismatch":maximum_error(*models2),
                          "ordinary_restore_ms":ordinary_restore_ms}),flush=True)

        # Strengthen the recovery test: the checkpoint must include warmed
        # Adam optimizer moments from a genuine earlier gradient step.
        warm=group.update(batch=batch,async_update=False)
        assert len(warm)==2
        warm_checkpoint=copy.deepcopy(group.get_state())
        warm_models=all_model_states(group)
        warm_optimizer=optimizer_numeric_leaves(warm_checkpoint)
        warm_peer_opt=all_optimizer_states(group)
        print("WARM_ADAM_PEER_STATE",
              json.dumps({"peer_optimizer_max_error":optimizer_max_abs_difference(
                              warm_peer_opt[0],warm_peer_opt[1]),
                          "leader_to_checkpoint_max_error":optimizer_max_abs_difference(
                              warm_optimizer,warm_peer_opt[0])}),flush=True)
        # Counterfactual: uninterrupted continuation from the warm checkpoint.
        trace_ready=group.foreach_learner(install_batch_provenance)
        assert len(trace_ready)==2 and all(r.ok and r.get() for r in trace_ready)
        healthy_continuation=group.update(batch=batch,async_update=False)
        assert len(healthy_continuation)==2
        reference_next_models=all_model_states(group)
        reference_next_optimizer=optimizer_numeric_leaves(group.get_state())
        reference_batch_trace=traced_batches(group)
        print("HEALTHY_STEP_ACTUAL_BATCH_SHA256",
              json.dumps(reference_batch_trace,sort_keys=True),flush=True)
        reference_peer_optimizer=all_optimizer_states(group)
        print("HEALTHY_WARM_CONTINUATION",
              json.dumps({"model_peer_error":maximum_error(*reference_next_models),
                          "optimizer_peer_error":optimizer_max_abs_difference(
                              reference_peer_optimizer[0],reference_peer_optimizer[1])}),
              flush=True)
        assert maximum_error(warm_models[0],reference_next_models[0])>0
        # A checkpoint older than the active weight sequence is NOT safely
        # rewound into the same live Learner via ordinary set_state: version
        # admission may decline the older weights, while optimizer state may
        # still be installed. A crash-recovery comparison must use a *fresh*
        # LearnerGroup. We therefore leave the healthy continuation intact,
        # kill one worker and compare a newly reconstructed group.
        print("WARM_ADAM_REFERENCE_CONTINUATION",
              json.dumps({"optimizer_buffers":len(warm_optimizer),
                          "reference_update_delta":maximum_error(
                              warm_models[0],reference_next_models[0])}),flush=True)
        baseline=warm_checkpoint
        models0=warm_models

        # Under normal operation, require *fresh*, identity-associated
        # acknowledgements, not only a cached healthy count.
        ids=manager.actor_ids()
        live=fresh_membership_attestation(group,ids)
        assert live["acks"]==2
        print("FRESH_GROUP_PREPARE_ACKS",json.dumps(live),flush=True)

        # True Ray actor termination: health can be stale until a fresh RPC.
        target=manager._actors[ids[0]]
        ray.kill(target,no_restart=True)
        killed=True
        # ray.kill requests actor termination, but a concurrently issued RPC
        # can still be accepted before the process actually exits. Synchronize
        # on a DIRECT Ray actor RPC, not via the manager, so cached health
        # remains unchanged until the membership challenge itself runs.
        death_observed=False
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            try:
                ray.get(target.apply.remote(lambda actor: True),timeout=2)
            except ray.exceptions.RayActorError:
                death_observed=True
                break
            except ray.exceptions.GetTimeoutError:
                pass
            time.sleep(.02)
        assert death_observed,"Ray killed actor but no observed death within deadline"
        print("DIRECT_ACTOR_DEATH_CONFIRMED_BEFORE_MEMBERSHIP_CHECK",flush=True)
        count_at_loss=manager.num_outstanding_async_reqs()
        healthy_before_ping=manager.num_healthy_actors()
        print("KILLED_ACTOR_BEFORE_LIVENESS_PROBE",
              json.dumps({"outstanding":count_at_loss,
                          "healthy_cached":healthy_before_ping,
                          "total":manager.num_actors()}),flush=True)
        assert count_at_loss==0
        # Cached health+empty queue appear safe, but a NEW participation
        # challenge MUST fail and prevent attempting checkpoint publication.
        assert healthy_before_ping==manager.num_actors(),healthy_before_ping
        try:
            fresh_membership_attestation(group,ids)
        except RuntimeError as error:
            assert "fresh membership ACK failed" in str(error),str(error)
            print("FRESH_GROUP_ACK_FENCED_DEAD_ACTOR",str(error)[:350],flush=True)
        else:
            raise AssertionError("dead Ray actor erroneously authenticated")

        # Send another real manager RPC (despite health being marked failed)
        # to confirm the actor-loss error is independently observable.
        calls=group.foreach_learner(lambda learner: learner.get_state(),
                                    healthy_only=False,timeout_seconds=12)
        failures=[str(call.get())[:240] for call in calls if not call.ok]
        assert failures,"actor loss not detected by real RPC"
        count_after_ping=manager.num_healthy_actors()
        assert count_after_ping<manager.num_actors(),(count_after_ping,failures)
        print("PHYSICAL_ACTOR_LOSS_DETECTED",
              json.dumps({"cached_health_before_rpc":healthy_before_ping,
                          "healthy_after_fresh_rpc":count_after_ping,
                          "failure_sample":failures[0]}),flush=True)
        assert manager.num_outstanding_async_reqs()==0

        # Recover after real physical Learner death by recreating the entire
        # DDP LearnerGroup, not replaying a checkpoint into a partial group.
        recovery_start=time.perf_counter()
        group.shutdown()
        group=None
        replacement=config.build_learner_group(env=env)
        group=replacement
        assert replacement._worker_manager.num_actors()==2
        replacement.set_state(copy.deepcopy(baseline))
        rebuild_and_restore_ms=(time.perf_counter()-recovery_start)*1000
        rebuilt_opt=optimizer_numeric_leaves(replacement.get_state())
        restored_opt_error=checked_optimizer_error(warm_optimizer,rebuilt_opt)
        print("WARM_ADAM_REBUILD_STATE_EQUIVALENCE",
              json.dumps({"optimizer_buffers":len(rebuilt_opt),
                          "optimizer_restore_max_error":restored_opt_error}),flush=True)
        replaced_models=all_model_states(replacement)
        recovered_peer_optimizer=all_optimizer_states(replacement)
        print("REBUILT_WARM_ADAM_PEER_STATE",
              json.dumps({"peer_optimizer_max_error":optimizer_max_abs_difference(
                              recovered_peer_optimizer[0],recovered_peer_optimizer[1]),
                          "leader_to_warm_max_error":optimizer_max_abs_difference(
                              warm_optimizer,recovered_peer_optimizer[0])}),flush=True)
        assert maximum_error(*replaced_models)<1e-6
        recovered_err=maximum_error(models0[0],replaced_models[0])
        assert recovered_err<1e-5,recovered_err
        print("PHYSICAL_CRASH_GROUP_REBUILT",
              json.dumps({"new_actor_count":2,"restored_max_error":recovered_err,
                          "peer_mismatch":maximum_error(*replaced_models),
                          "rebuild_and_restore_ms":rebuild_and_restore_ms,
                          "rebuild_over_steady_restore":
                          rebuild_and_restore_ms/ordinary_restore_ms}),flush=True)
        # Require a genuine PyTorch DDP gradient to succeed *after* rebuild.
        trace_ready=replacement.foreach_learner(install_batch_provenance)
        assert len(trace_ready)==2 and all(r.ok and r.get() for r in trace_ready)
        return_values=replacement.update(batch=batch,async_update=False)
        recovered_batch_trace=traced_batches(replacement)
        print("RECOVERED_STEP_ACTUAL_BATCH_SHA256",
              json.dumps(recovered_batch_trace,sort_keys=True),flush=True)
        print("ACTUAL_BATCH_PROVENANCE_MATCHES",
              json.dumps({"same_per_rank":reference_batch_trace==recovered_batch_trace}),
              flush=True)
        assert len(return_values)==2,return_values
        trained=all_model_states(replacement)
        training_delta=maximum_error(replaced_models[0],trained[0])
        assert training_delta>0,training_delta
        diagnostic_peer_error=maximum_error(*trained)
        trajectory_error=maximum_error(reference_next_models[0],trained[0])
        current_optimizer=all_optimizer_states(replacement)
        print("POST_CRASH_GRADIENT_TRAJECTORY_DIAGNOSTICS",
              json.dumps({"reference_model_peer_error":maximum_error(*reference_next_models),
                          "recovered_model_peer_error":diagnostic_peer_error,
                          "reference_vs_recovered_model_error":trajectory_error,
                          "recovered_optimizer_peer_error":optimizer_max_abs_difference(
                              current_optimizer[0],current_optimizer[1]),
                          "recovered_optimizer_vs_reference_error":optimizer_max_abs_difference(
                              reference_next_optimizer,current_optimizer[0])}),flush=True)
        assert diagnostic_peer_error<1e-5,diagnostic_peer_error
        assert trajectory_error<1e-5,trajectory_error
        recovered_optimizer=optimizer_numeric_leaves(replacement.get_state())
        optimizer_trajectory_error=checked_optimizer_error(
            reference_next_optimizer,recovered_optimizer)
        print("WARM_ADAM_POST_CRASH_CONTINUATION_EQUIVALENCE",
              json.dumps({"model_next_step_error":trajectory_error,
                          "optimizer_next_step_error":optimizer_trajectory_error,
                          "optimizer_buffers":len(recovered_optimizer)}),flush=True)
        print("RECOVERED_TWO_DDP_LEARNERS_TRAINED",
              json.dumps({"batch_rows":batch.count,"max_update":training_delta,
                          "post_gradient_peer_mismatch":maximum_error(*trained)}),
              flush=True)
        print("TWO_LEARNER_ASYNC_LOSS_CONTRACT_PASSED",flush=True)
    finally:
        if group is not None:
            try:
                group.shutdown()
            except Exception as exc:
                print("SHUTDOWN_AFTER_ACTOR_LOSS",type(exc).__name__,str(exc)[:160],
                      flush=True)
        env.close()
        ray.shutdown()


if __name__=="__main__":
    main()
