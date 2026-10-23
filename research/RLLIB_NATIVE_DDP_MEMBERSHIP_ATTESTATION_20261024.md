# Real two-Learner DDP: asynchronous quiescence, physical actor loss, and checkpoint membership

**Date:** 2026-10-24 (Asia/Singapore). **Maturity:** fork-only Ray 2.59.0 CPU research. This is not an official upstream production patch or a previously unpublished consensus protocol.

## Core finding: cached health is not a commit participation proof

Official Ray `LearnerGroup.update(async_update=True)` has a dedicated request queue managed by `FaultTolerantActorManager`, and `LearnerGroup.set_state` independently dispatches checkpoint state to Learners. The manager exposes `num_outstanding_async_reqs`, `num_healthy_actors`, and `foreach_actor`. Healthy state is maintained by observations of prior actor communications and can remain stale after physical actor loss.

**Native reproducible witness** — [fully green two-Learner DDP CI #40004022632](https://github.com/lindicaphxag-tech/ray/actions/runs/40004022632):
- Built **real native `LearnerGroup(num_learners=2)`**, CPU PyTorch distributed backend and CartPole batch, through unmodified Ray 2.59.0 API.
- Initial two RLModule model tensor snapshots matched; dispatched a true `update(async_update=True)` operation to each Learner (2 outstanding actor tasks). Both finished successfully, real gradient max abs delta **0.0010000020265579224**, peer weight mismatch **0**.
- Drained both via the actual `FaultTolerantActorManager.fetch_ready_async_reqs` without accidentally dispatching more training. A native `LearnerGroup.set_state` from the original complete checkpoint restored both: max abs error **0**, peer mismatch **0**.
- Physically killed **one** Learner actor using `ray.kill(handle,no_restart=True)`, *after* the async queue was drained. Without any intervening manager RPC, measurements were **outstanding=0**, **cached healthy=2**, **total=2**. A fresh manager RPC reported the dead actor and refreshed healthy to **1**.

This is an actual actor-death counterexample rather than `set_actor_state(...,False)` metadata injection.

### Indistinguishability argument (restricted to cached counters)

Consider executions A (two actors alive, idle) and B (two actors were alive and idle, then one died without any observed Ray RPC). At a decision time before the next manager communication, the coordinator sees identical cached pair `(outstanding=0, cached_healthy=2)` in both. A deterministic acceptance predicate depending **only** on those counters therefore produces the same decision in both, even though one has lost a required checkpoint participant.

This is a direct instance of stale failure-detector observations, **not a new distributed computing impossibility theorem**. It falsifies this particular sufficient admission claim:

```
num_outstanding_async_reqs()==0 and num_healthy_actors()==num_actors()
    => all expected actor participants are currently alive
```

### Fresh challenge contrast

[Second green native DDP run #40004378668](https://github.com/lindicaphxag-tech/ray/actions/runs/40004378668) added a per-checkpoint nonce challenge sent to **all actor IDs** (including currently marked unhealthy). While both were alive: **2 unique actor-ID replies with the correct nonce**. After physically killing one actor, manager health still said 2/2; new challenge obtained a Ray actor-death error and **refused the checkpoint**. On the following independent liveness RPC the manager reported only one healthy actor.

This is a point-in-time liveness/participation check, **not** a proof of continuously surviving membership, secure challenge-response authentication or an atomic commit. An actor can still die *after* successful challenge; correctness requires retaining a fence, associating stage ACKs with a specific checkpoint epoch/actor identity, and failing closed on missing participants.

## A scoped candidate high-integrity group protocol

Under a *single mutually exclusive* group-level entry:
1. Close admissions to new async `LearnerGroup.update` tasks (and prevent all out-of-band pathways from bypassing this admission).
2. Drain every existing update from `FaultTolerantActorManager`, inspecting errors and actual membership. Never drain by repeatedly calling `LearnerGroup.update(async_update=True)` because that resubmits work.
3. Bind the expected actor IDs and a new checkpoint epoch/challenge; obtain fresh `PREPARED` ACKs from every required actor after staging the full checkpoint **while all training stays fenced**.
4. Expose the new epoch to training and EnvRunners only after all required actors have provided matching ACKs. A duplicate/stale/delayed ACK must not count for another epoch.
5. If a required actor disappears, fails to stage any component, or responds too late, abort candidate state, **rebuild the entire LearnerGroup from the last authoritative complete snapshot**, verify model+optimizer+selector, and only then reopen submissions.

The program currently verifies separate pieces of this outline in native experiment harnesses, but **does not implement this integrated transactional API in Ray production**. In particular the two-phase stage/commit plus fresh membership ACK are not one atomic production operation.

## Correctness and efficiency context

- [One-remote-Learner actual async-group drain + unhealthy metadata refusal](https://github.com/lindicaphxag-tech/ray/actions/runs/40002895154).
- [Native two-actor coordinator with controlled setter fault, withheld ACK, delayed commit and model reconstruction](https://github.com/lindicaphxag-tech/ray/actions/runs/39993254415).
- [Actual native Learner.update overlapping checkpoint initiation](https://github.com/lindicaphxag-tech/ray/actions/runs/39997227527).
- [Native per-update vs batched remote lease benchmark](https://github.com/lindicaphxag-tech/ray/actions/runs/40002166087): 16 actual CPU gradient steps (4 rotated blocks) 28.64ms without remote lease, 70.73ms per-step RPC lease (2.47x), 34.72ms with K=8 amortization (1.21x). No GPU or reward inference.

Group-level quiescence is motivated by correct *checkpoint* admission while avoiding high per-small-step RPC cost. Bounds on pause duration, unknown training entry points, actor crashes during actual state staging, and process crash durability remain unsolved.

## Decision and explicit gaps

A fully safe and reviewable upstream proposal still requires:
- one exclusive group-level gate that all native update/set-state and actor-manager scheduling paths honor;
- stale/missing epoch-ACK rejection and crash-after-ACK failure injection, not just a one-shot ping;
- full authoritative state recovery after physically killed Learner actors (model/optimizer/selector/RNG as applicable), and an actual subsequent DDP gradient step;
- held in-flight leases or draining semantics covering real async actor work, including task exceptions and canceled/dropped tasks;
- comparative recovery and throughput trials, GPU/DDP scaling where feasible, and maintainer API agreement.

**Never count these as an upstream Merge or as a proved novel distributed algorithm.**

### Reproduction

[Fork research PR #11](https://github.com/lindicaphxag-tech/ray/pull/11) keeps all experiments in a single Draft review branch; the native 2-Learner actor-loss test is `ci/rllib_native_two_learner_ddp_crash_health.py` and its Actions workflow is `.github/workflows/rllib_native_two_learner_ddp_crash_health.yml`.


## Completed physical-crash recovery with resumed native distributed gradient

[Final native CI #40004680729](https://github.com/lindicaphxag-tech/ray/actions/runs/40004680729) **passed** the full expanded experiment, including all of the above participation checks and a real actor kill.

After the liveness challenge detected one physically dead learner, the test stopped the old `LearnerGroup`, constructed a **new real 2-Learner CPU DDP group** from the original `BaseTestingAlgorithmConfig`, and applied the last-known-authoritative group checkpoint with the actual `LearnerGroup.set_state` API. The new group had 2 participants; both recovered models matched the authoritative weights with maximum absolute error **0**, and peer mismatch was **0**. The rebuilt group then completed a **new synchronous real 200-row CartPole distributed gradient**, max tensor change **0.0010000020265579224** and post-update cross-Learner mismatch **0**.

This is materially stronger than a source-only mocked recovery: the killed Ray process was gone, both new native DDP workers were actually initialized, the real checkpoint was installed and real gradients subsequently ran.

**Scope:** the coordinator was alive throughout, the checkpoint remained available in the parent Python process, and the entire group was rebuilt. Not tested: automatic production failover, persistent checkpoint durability, actor failures *during* the apply/commit stage, model/optimizer state checksum beyond inspected tensor state, Ray placement groups across real machines, malicious/non-responsive nodes, or the reward/throughput tradeoff under natural training workloads. Group destruction and creation may take seconds; this test is an existence/correctness witness, not proof of a fast low-disruption recovery architecture.

**Decision:** Before submitting any new upstream production implementation, discuss with maintainers whether the right scope is an opt-in `LearnerGroup` commit/quiescence method incorporating membership ACKs and fail-closed rebuilding. Existing `#67131` and `#67329` should remain narrow correctness fixes and must not absorb this unreviewed distributed feature.
