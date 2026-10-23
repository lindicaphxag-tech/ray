"""Native two-Learner RLlib asynchronous training / stale actor-health witness.

Native Ray 2.59, torch CPU DDP; no production modifications. Demonstrates that
the async queue and health metadata are not a fresh proof of actor liveness.
The loss probe is after one successful distributed update and restore.
"""
from __future__ import annotations

import copy
import json
import secrets
import time

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader


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
        group.set_state(copy.deepcopy(baseline))
        models2=all_model_states(group)
        assert maximum_error(*models2)<1e-6
        restore_error=maximum_error(models0[0],models2[0])
        assert restore_error<1e-5,restore_error
        print("TWO_DDP_LEARNERS_RESTORED",
              json.dumps({"restored_max_error":restore_error,
                          "peer_mismatch":maximum_error(*models2)}),flush=True)

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
        group.shutdown()
        group=None
        replacement=config.build_learner_group(env=env)
        group=replacement
        assert replacement._worker_manager.num_actors()==2
        replacement.set_state(copy.deepcopy(baseline))
        replaced_models=all_model_states(replacement)
        assert maximum_error(*replaced_models)<1e-6
        recovered_err=maximum_error(models0[0],replaced_models[0])
        assert recovered_err<1e-5,recovered_err
        print("PHYSICAL_CRASH_GROUP_REBUILT",
              json.dumps({"new_actor_count":2,"restored_max_error":recovered_err,
                          "peer_mismatch":maximum_error(*replaced_models)}),flush=True)
        # Require a genuine PyTorch DDP gradient to succeed *after* rebuild.
        return_values=replacement.update(batch=batch,async_update=False)
        assert len(return_values)==2,return_values
        trained=all_model_states(replacement)
        training_delta=maximum_error(replaced_models[0],trained[0])
        assert training_delta>0,training_delta
        assert maximum_error(*trained)<1e-5,maximum_error(*trained)
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
