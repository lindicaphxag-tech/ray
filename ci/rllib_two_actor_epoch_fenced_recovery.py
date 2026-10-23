"""Two native Ray Learner actors: fenced checkpoint publication and recovery.

This is a coordinator-level RESEARCH PROTOCOL, not a production Ray patch or
cluster-wide linearizability proof. Tests the reviewed production
Learner.set_state method on actual Ray 2.59.0 actor processes / torch states.
During staging, ALL train/update methods exposed by this test wrapper refuse
work. Failure forces rebuilding BOTH from the authoritative last commit.

The wrapper cannot prevent access to the raw Learner from code outside it,
and no claim is made about real crash durability or GPU training.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import pathlib
import sys
import time
from contextlib import suppress
from unittest.mock import patch
from ray.rllib.utils.test_utils import get_cartpole_dataset_reader

import gymnasium as gym
import numpy as np
import ray
import ray.rllib.core.learner.learner as learner_module
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.testing.testing_learner import BaseTestingAlgorithmConfig
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO


def patch_real_set_state(source: str) -> None:
    """Replace the installed wheel's method with exactly the review branch AST."""
    filename=pathlib.Path(source)
    tree=ast.parse(filename.read_text(),filename=str(filename))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Learner")
    method=copy.deepcopy(next(n for n in klass.body
                               if isinstance(n,ast.FunctionDef) and n.name=="set_state"))
    method.decorator_list=[]
    method.returns=None
    for a in [*method.args.args,*method.args.kwonlyargs]:
        a.annotation=None
    scope=dict(vars(learner_module))
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[method],type_ignores=[])),str(filename),"exec"),scope)
    learner_module.Learner.set_state=scope["set_state"]


def module_fingerprint(module_state):
    h=hashlib.sha256()
    # A parameter identity check, NOT cryptographic attestation of a remote server.
    for mid, values in sorted(module_state.items()):
        h.update(str(mid).encode())
        for name,val in sorted(values.items()):
            arr=np.asarray(val)
            h.update(str(name).encode())
            h.update(arr.dtype.str.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def mutate_first_weight(module_state):
    out=copy.deepcopy(module_state)
    first=out[DEFAULT_MODULE_ID]
    for name,item in first.items():
        arr=np.asarray(item)
        if arr.size>0 and np.issubdtype(arr.dtype,np.floating):
            first[name]=arr.copy()+np.asarray(.125,dtype=arr.dtype)
            return out
    raise AssertionError("no float parameter in native policy")


@ray.remote(num_cpus=0)
class EpochAuthority:
    """Single-authority read lease; deliberately not process-crash durable."""
    def __init__(self):
        self.live_epoch=None
        self.pending=None
        self.fenced=True
        self.active={}
        self.inside_gradient=set()
        self.next_ticket=0

    def acquire(self,epoch):
        if self.fenced or self.live_epoch!=epoch:
            return None
        self.next_ticket+=1
        self.active[self.next_ticket]=epoch
        return self.next_ticket

    def gradient_entered(self,ticket):
        assert ticket in self.active
        self.inside_gradient.add(ticket)

    def release(self,ticket):
        assert ticket in self.active,ticket
        self.inside_gradient.discard(ticket)
        del self.active[ticket]

    def bootstrap(self,epoch):
        assert self.live_epoch is None
        self.live_epoch=epoch

    def begin(self,new_epoch):
        assert not self.fenced and self.pending is None
        if self.active:
            raise RuntimeError("in-flight training lease prevents new epoch")
        assert new_epoch>self.live_epoch
        self.fenced=True
        self.pending=new_epoch
        return self.snapshot()

    def abort(self,new_epoch):
        assert self.pending==new_epoch
        self.pending=None
        self.fenced=True
        return self.snapshot()

    def release_recovered(self,epoch):
        assert self.pending is None and self.live_epoch==epoch
        assert not self.active
        self.fenced=False
        return self.snapshot()

    def publish(self,epoch,acks):
        assert self.pending==epoch and len(acks)==2
        assert not self.active
        assert all(x["status"]=="PREPARED" and x["snapshot"]["local_version"]==epoch
                   for x in acks),acks
        self.live_epoch=epoch
        self.pending=None
        self.fenced=False
        return self.snapshot()

    def can_train(self,epoch):
        return not self.fenced and self.live_epoch==epoch

    def snapshot(self):
        return {"live_epoch":self.live_epoch,"pending":self.pending,
                "fenced":self.fenced,"inflight":len(self.active),
                "inside_gradient":len(self.inside_gradient)}


@ray.remote(num_cpus=1)
class FencedNativeLearner:
    def __init__(self,source,rank,authority):
        patch_real_set_state(source)
        self.rank=rank
        self.authority=authority
        self.env=gym.make("CartPole-v1")
        self.learner=BaseTestingAlgorithmConfig().build_learner(env=self.env)
        self.epoch=None
        self.staged=None
        self.quarantined=True
        self.train_allowed=False
        self.held_ticket=None

    def initial_snapshot(self):
        snap=self.learner.get_state()
        snap[WEIGHTS_SEQ_NO]=1
        snap["should_module_be_updated"]={DEFAULT_MODULE_ID}
        return snap

    def _fresh(self,state,epoch):
        # Rebuild from authoritative state; do not rely on damaged in-place
        # objects or replay a checkpoint at the currently published version.
        self.learner=BaseTestingAlgorithmConfig().build_learner(env=self.env)
        self.learner.set_state(copy.deepcopy(state))
        current=self.learner.get_state()
        assert current[WEIGHTS_SEQ_NO]==epoch,(current[WEIGHTS_SEQ_NO],epoch)
        self.epoch=epoch
        self.staged=None
        self.quarantined=True
        self.train_allowed=False
        return self.status()

    def restore(self,state,epoch):
        return self._fresh(state,epoch)

    def status(self):
        snap=self.learner.get_state()
        return {"actor":self.rank,"local_version":int(snap[WEIGHTS_SEQ_NO]),
                "committed_epoch":self.epoch,"staged":self.staged,
                "quarantined":self.quarantined,"train_allowed":self.train_allowed,
                "selector":repr(snap["should_module_be_updated"]),
                "default_trainable":bool(self.learner.should_module_be_updated(DEFAULT_MODULE_ID)),
                "weights_hash":module_fingerprint(snap[COMPONENT_RL_MODULE])}

    def fence(self):
        self.train_allowed=False
        self.quarantined=True
        return self.status()

    def stage(self,state,epoch,inject_selector_failure=False):
        assert not self.train_allowed and self.quarantined,"writer did not fence reader"
        assert self.epoch is not None and epoch>self.epoch
        try:
            if inject_selector_failure:
                original=self.learner.config.multi_agent
                def fail(*, policies_to_train):
                    raise RuntimeError("injected selector commit failure")
                self.learner.config.multi_agent=fail
            try:
                self.learner.set_state(copy.deepcopy(state))
            finally:
                if inject_selector_failure:
                    self.learner.config.multi_agent=original
        except RuntimeError as e:
            self.staged=None
            self.train_allowed=False
            self.quarantined=True
            return {"status":"FAIL","actor":self.rank,"error":str(e),
                    "snapshot":self.status()}
        self.staged=epoch
        return {"status":"PREPARED","actor":self.rank,"snapshot":self.status()}

    def admit(self,epoch):
        assert self.quarantined and not self.train_allowed
        assert self.staged==epoch,(self.rank,self.staged,epoch)
        actual=self.learner.get_state()
        assert actual[WEIGHTS_SEQ_NO]==epoch
        # Only called after ALL peers ACK: scoped single-coordinator gate.
        self.epoch=epoch
        self.staged=None
        self.quarantined=False
        self.train_allowed=True
        return self.status()

    def resume_recovered(self,epoch):
        assert self.quarantined and self.epoch==epoch and self.staged is None
        self.quarantined=False
        self.train_allowed=True
        return self.status()

    def hold_training_lease(self):
        assert self.train_allowed and not self.quarantined
        assert self.held_ticket is None
        ticket=ray.get(self.authority.acquire.remote(self.epoch))
        assert ticket is not None, "authority rejected training lease"
        self.held_ticket=ticket
        return {"actor":self.rank,"ticket":ticket}

    def end_training_lease(self):
        assert self.held_ticket is not None
        ray.get(self.authority.release.remote(self.held_ticket))
        self.held_ticket=None
        return {"actor":self.rank,"result":"LEASE_RELEASED"}

    def real_gradient_update_under_lease(self):
        """Exercise actual native Learner.update while holding the central lease.

        A brief pause inside the native before_gradient_based_update hook makes
        the interleaving reproducible; parameter updates still run normally.
        """
        assert self.train_allowed and not self.quarantined
        assert self.learner.should_module_be_updated(DEFAULT_MODULE_ID)
        ticket=ray.get(self.authority.acquire.remote(self.epoch))
        assert ticket is not None
        try:
            before=self.learner.get_state()[COMPONENT_RL_MODULE][DEFAULT_MODULE_ID]
            before={k:np.asarray(v).copy() for k,v in before.items()
                    if np.issubdtype(np.asarray(v).dtype,np.floating)}
            reader=get_cartpole_dataset_reader(batch_size=32)
            batch=self.learner._convert_batch_type(reader.next().as_multi_agent())
            calls=[0]
            def hook(*, timesteps):
                calls[0]+=1
                ray.get(self.authority.gradient_entered.remote(ticket))
                time.sleep(1.0)
            with patch.object(self.learner,"before_gradient_based_update",side_effect=hook):
                self.learner.update(batch=batch)
            assert calls[0]>=1,"native before-gradient hook was not called"
            after=self.learner.get_state()[COMPONENT_RL_MODULE][DEFAULT_MODULE_ID]
            differences=[float(np.max(np.abs(np.asarray(after[k])-v)))
                         for k,v in before.items()]
            max_change=max(differences)
            assert max_change>0,max_change
            return {"actor":self.rank,"rows":batch.count,
                    "max_param_change":max_change,"hook_calls":calls[0]}
        finally:
            ray.get(self.authority.release.remote(ticket))

    def train_probe(self):
        # Every wrapper-mediated train call holds a central ticket while
        # making its policy decision. Real Learner.update() is not yet
        # instrumented here; hold_training_lease simulates its in-flight span.
        ticket=ray.get(self.authority.acquire.remote(self.epoch))
        if ticket is None:
            return {"actor":self.rank,"result":"AUTHORITY_FENCED"}
        try:
            if not self.train_allowed or self.quarantined:
                return {"actor":self.rank,"result":"BLOCKED"}
            if not self.learner.should_module_be_updated(DEFAULT_MODULE_ID):
                return {"actor":self.rank,"result":"SELECTOR_BLOCKED"}
            return {"actor":self.rank,"result":"ELIGIBLE"}
        finally:
            ray.get(self.authority.release.remote(ticket))

    def close(self):
        with suppress(Exception):
            self.env.close()


def all_status(actors):
    return ray.get([actor.status.remote() for actor in actors])


def assert_same(actors,epoch,allow_train):
    states=all_status(actors)
    assert all(x["committed_epoch"]==epoch and
               x["local_version"]==epoch and
               x["train_allowed"]==allow_train for x in states), states
    hashes={x["weights_hash"] for x in states}
    assert len(hashes)==1,states
    return states


def run():
    source=str(pathlib.Path(sys.argv[1]).resolve())
    ray.init(num_cpus=4,include_dashboard=False,ignore_reinit_error=True,
             log_to_driver=False)
    authority=EpochAuthority.remote()
    actors=[FencedNativeLearner.remote(source,i,authority) for i in range(2)]
    try:
        authoritative=ray.get(actors[0].initial_snapshot.remote())
        start=time.perf_counter()
        first=ray.get([a.restore.remote(authoritative,1) for a in actors])
        assert all(not x["train_allowed"] for x in first)
        ray.get(authority.bootstrap.remote(1))
        first=ray.get([a.resume_recovered.remote(1) for a in actors])
        ray.get(authority.release_recovered.remote(1))
        assert_same(actors,1,True)
        assert all(x["result"]=="ELIGIBLE" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        setup_ms=(time.perf_counter()-start)*1000
        print("INITIAL_COMMIT",json.dumps({"time_ms":setup_ms,"state":first}),flush=True)

        intended=copy.deepcopy(authoritative)
        intended[WEIGHTS_SEQ_NO]=2
        intended[COMPONENT_RL_MODULE]=mutate_first_weight(authoritative[COMPONENT_RL_MODULE])
        intended["should_module_be_updated"]={"blocked_other_policy"}

        # Actual Ray Learner.update in actor 0 acquires a central ticket.
        # Wait for the *native* gradient hook, then concurrently attempt a
        # checkpoint publish while the real update is paused inside its hook.
        future=actors[0].real_gradient_update_under_lease.remote()
        deadline=time.monotonic()+35
        while time.monotonic()<deadline:
            gate=ray.get(authority.snapshot.remote())
            if gate["inside_gradient"]>0:
                break
            if ray.wait([future],timeout=0)[0]:
                ray.get(future)  # surface a native Ray/gradient failure
                raise AssertionError("real update completed before overlap hook")
            time.sleep(0.025)
        else:
            raise AssertionError("native gradient hook never entered")
        try:
            ray.get(authority.begin.remote(2))
        except ray.exceptions.RayTaskError as exc:
            assert "in-flight training lease" in str(exc),str(exc)
        else:
            raise AssertionError("checkpoint began while real gradient was executing")
        gradient_result=ray.get(future)
        assert gradient_result["max_param_change"]>0
        assert ray.get(authority.snapshot.remote())["inflight"]==0
        # Genuine policy parameters changed; rebuild the authoritative e1
        # checkpoint to leave the existing cross-actor protocol probes intact.
        ray.get(actors[0].restore.remote(authoritative,1))
        ray.get(actors[0].resume_recovered.remote(1))
        assert_same(actors,1,True)
        print("REAL_GRADIENT_LEASE_FENCED",json.dumps(gradient_result),flush=True)

        # Hold an in-flight training lease while the coordinator attempts
        # an update: begin() MUST refuse instead of racing with a reader.
        ticket=ray.get(actors[0].hold_training_lease.remote())
        assert ticket["ticket"]>0
        try:
            ray.get(authority.begin.remote(2))
        except ray.exceptions.RayTaskError as err:
            assert "in-flight training lease" in str(err),str(err)
        else:
            raise AssertionError("authority began an epoch during in-flight training")
        assert ray.get(authority.snapshot.remote())["inflight"]==1
        ray.get(actors[0].end_training_lease.remote())
        assert ray.get(authority.snapshot.remote())["inflight"]==0
        print("TRAINING_LEASE_DRAIN: rejected begin until in-flight ticket released",
              flush=True)

        t0=time.perf_counter()
        ray.get(authority.begin.remote(2))
        # Even BEFORE local fences arrive, authority rejects training.
        assert all(x["result"]=="AUTHORITY_FENCED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        ray.get([a.fence.remote() for a in actors])
        assert all(x["result"]=="AUTHORITY_FENCED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        ack=ray.get([actors[0].stage.remote(intended,2),
                     actors[1].stage.remote(intended,2,inject_selector_failure=True)])
        assert [x["status"] for x in ack]==["PREPARED","FAIL"],ack
        witness=ack[1]["snapshot"]
        assert witness["local_version"]==2 and witness["committed_epoch"]==1,witness
        assert witness["default_trainable"] is True,witness
        assert all(x["result"]=="AUTHORITY_FENCED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        ray.get(authority.abort.remote(2))
        t1=time.perf_counter()
        print("FAILURE_FENCED",json.dumps({"acks":ack,"stage_ms":(t1-t0)*1000}),flush=True)

        # One failed actor poisons the candidate epoch: rebuild every peer
        # from the LAST AUTHORITATIVELY COMMITTED state (epoch 1).
        restored=ray.get([a.restore.remote(authoritative,1) for a in actors])
        assert_same(actors,1,False)
        assert all(x["result"]=="AUTHORITY_FENCED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        ray.get([a.resume_recovered.remote(1) for a in actors])
        restored_live=assert_same(actors,1,True)
        assert all(x["result"]=="AUTHORITY_FENCED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        ray.get(authority.release_recovered.remote(1))
        assert all(x["result"]=="ELIGIBLE" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        recovery_ms=(time.perf_counter()-t1)*1000
        print("RECOVERED_EPOCH_1",json.dumps({"recovery_ms":recovery_ms,
                                            "state":restored_live}),flush=True)

        # Retry the newer complete epoch after authoritative rebuild.
        # Acknowledge loss: both workers have installed, but coordinator
        # receives only one ACK. It MUST abort, not publish.
        ray.get(authority.begin.remote(2))
        ray.get([a.fence.remote() for a in actors])
        missing_ack=ray.get([a.stage.remote(intended,2) for a in actors])
        assert all(x["status"]=="PREPARED" for x in missing_ack),missing_ack
        assert all(x["result"]=="AUTHORITY_FENCED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        ray.get(authority.abort.remote(2))
        ray.get([a.restore.remote(authoritative,1) for a in actors])
        assert_same(actors,1,False)
        ray.get([a.resume_recovered.remote(1) for a in actors])
        ray.get(authority.release_recovered.remote(1))
        assert_same(actors,1,True)
        print("MISSING_ACK_ABORT correctly recovered both installed actors to e1",
              flush=True)

        t2=time.perf_counter()
        ray.get(authority.begin.remote(2))
        ray.get([a.fence.remote() for a in actors])
        good=ray.get([a.stage.remote(intended,2) for a in actors])
        assert all(x["status"]=="PREPARED" for x in good),good
        assert all(x["result"]=="AUTHORITY_FENCED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        # Global epoch publication only after verifying both stage ACKs.
        ray.get(authority.publish.remote(2,good))
        # Both local epochs are still e1; authority now grants only e2.
        assert all(x["result"]=="AUTHORITY_FENCED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        # Delayed local commit notification: first actor has e2 lease while
        # second remains locally fenced; second cannot train prematurely.
        one=ray.get(actors[0].admit.remote(2))
        delayed=ray.get(actors[1].train_probe.remote())
        assert delayed["result"]=="AUTHORITY_FENCED",delayed
        assert ray.get(actors[0].train_probe.remote())["result"]=="SELECTOR_BLOCKED"
        committed=[one,ray.get(actors[1].admit.remote(2))]
        assert_same(actors,2,True)
        assert all(x["result"]=="SELECTOR_BLOCKED" for x in ray.get(
            [a.train_probe.remote() for a in actors]))
        success_ms=(time.perf_counter()-t2)*1000
        print("COMMITTED_EPOCH_2",json.dumps({"success_ms":success_ms,
                                            "state":committed}),flush=True)
        print("TWO-ACTOR NATIVE EPOCH FENCING AND RECOVERY PASSED",flush=True)
        return {"initial_ms":setup_ms,"fault_stage_ms":(t1-t0)*1000,
                "recovery_ms":recovery_ms,"retry_ms":success_ms}
    finally:
        ray.get([a.close.remote() for a in actors])
        ray.shutdown()


if __name__=="__main__":
    print("MEASUREMENTS",json.dumps(run()),flush=True)
