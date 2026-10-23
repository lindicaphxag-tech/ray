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

    def bootstrap(self,epoch):
        assert self.live_epoch is None
        self.live_epoch=epoch

    def begin(self,new_epoch):
        assert not self.fenced and self.pending is None
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
        self.fenced=False
        return self.snapshot()

    def publish(self,epoch,acks):
        assert self.pending==epoch and len(acks)==2
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
                "fenced":self.fenced}


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

    def train_probe(self):
        # This authorization check is made at the actual wrapper's read path.
        # Even a delayed/missing local fence cannot override authority fencing.
        if not ray.get(self.authority.can_train.remote(self.epoch)):
            return {"actor":self.rank,"result":"AUTHORITY_FENCED"}
        if not self.train_allowed or self.quarantined:
            return {"actor":self.rank,"result":"BLOCKED"}
        # A real Learner module eligibility gate, not a reward claim.
        if not self.learner.should_module_be_updated(DEFAULT_MODULE_ID):
            return {"actor":self.rank,"result":"SELECTOR_BLOCKED"}
        return {"actor":self.rank,"result":"ELIGIBLE"}

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
