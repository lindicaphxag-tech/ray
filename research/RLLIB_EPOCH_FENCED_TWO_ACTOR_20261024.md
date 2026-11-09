# Epoch-fenced RLlib state publication — native two-actor research receipt

**Date:** 2026-10-24 Singapore time. **Status:** Fork-only research; not an official Ray production patch, not an accepted PR or a newly invented distributed commit protocol.

## Why version admission alone is insufficient

The previously demonstrated 3-component Learner guard protects *successful* `set_state` calls from combining rejected model weights with older Adam/selector data, but the real production method writes components sequentially. Injecting a selector setter exception can leave new weights and optimizer, an advanced version, and an old selector. The same-version packet cannot repair the state; a following 200-row CartPole step demonstrably updates the wrong policy.

- Native failing checkpoint + actual gradient step: [CI 39989496133](https://github.com/lindicaphxag-tech/ray/actions/runs/39989496133).
- Six source-method setter failures: [CI 39988732670](https://github.com/lindicaphxag-tech/ray/actions/runs/39988732670).
- [Ray issue #67292](https://github.com/ray-project/ray/issues/67292#issuecomment-6370064398).

## Two-native-actor prototype

The exact reviewed production `Learner.set_state` method, unchanged and compiled from the source AST, runs inside **two separate actual Ray 2.59.0 actor processes** with real PyTorch RLModules and optimizer state. A single coordinator governs all *test-wrapper* trainability probes and epoch publication.

Define:

- Authority: `(e_public, e_candidate, fenced)`
- Actor i: `(e_local, e_staged, ready, W_i, O_i, T_i)`
- Training admission within this wrapper: obtain a central **epoch lease ticket** while `ready_i ∧ ¬fenced ∧ (e_local_i = e_public)`. Release it on completion. The coordinator rejects `begin(new_epoch)` while any ticket is held.
- Begin update: authority fences **before** any per-actor stop signal; actors cannot train after that point *for a newly started wrapper-mediated operation*.
- Prepare: every actor installs complete `(W,O,T)` while locally fenced. The coordinator must collect `PREPARED` acknowledgements from both.
- Abort: if any stage fails or even one ACK is unavailable, **do not publish the candidate version**. Recreate **both Learners** from the prior authoritative complete checkpoint, verify their local weight fingerprints and epoch, then un-fence authority.
- Publish: only after both `PREPARED` ACKs have been checked does authority publish `e_candidate`. An actor with a delayed local `admit` does not yet match that epoch and remains unable to train.

This is a **two-process, single-coordinator fail-closed execution contract** built from established fenced publication and reconstruction techniques. It does not add atomic setter semantics to RLlib.

## Evidence

**Initial real two-actor result:** [Actions 39992394670](https://github.com/lindicaphxag-tech/ray/actions/runs/39992394670), successful.
- Epoch 1: two actors have matching model weight SHA-256 fingerprints; both eligible to train.
- Epoch 2: Actor 0 PREPARED, Actor 1 raises at selector setter, leaving version 2 and the old selector. Both are fenced.
- Recover: rebuild both from authoritative epoch 1; check matching fingerprints and selectors; then retry epoch 2 successfully.
- Measured coordinator operations (single run, runner-dependent): first staging + failure 3.46 ms; two-actor recovery 14.96 ms; successful retry 5.04 ms.

**Strengthened authority-gated result:** [Actions 39993254415](https://github.com/lindicaphxag-tech/ray/actions/runs/39993254415), **all steps successful**.
1. Central `begin(2)` rejects both actors' training probes **even before local fence messages**; authority-first ordering.
2. A genuine selector setter error occurs on actor 1. Actor 0 was prepared; actor 1 had new weights and a new local version but the old training selector. All wrapper-mediated training probes were rejected.
3. Both are reconstructed from e1, checked to have identical model hashes, and un-fenced safely.
4. Both actors successfully install e2, but the coordinator **deliberately withholds one prepared ACK**; it aborts and reconstructs both from authoritative e1. Neither e2 packet is published.
5. Both actors install e2 again and ACK. Authority publishes e2. Before local `admit` notifications, both probes are rejected with `AUTHORITY_FENCED` because they still advertise e1. After the first `admit`, actor 1 remains blocked until its own acknowledgement is applied.
6. After both local commits, the weights match and the model version is e2. The new selector excludes the default policy and the wrapper correctly reports `SELECTOR_BLOCKED` on both actors.

Single CI run wall times: initial startup-coordination 189.94 ms; fault stage + denial 11.56 ms; first authoritative recovery 35.63 ms; final successful publish 18.31 ms. **These are not replicated throughput measurements** and are not directly comparable across runs without a controlled fixture.

[Exact experiment script](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_two_actor_epoch_fenced_recovery.py)

## Invariants, scope and remaining counterexamples

**Checked under the scripted sequential coordinator and wrapper-mediated accesses:**
- If a stage fails or an ACK is lost, the candidate epoch is not published and both Actors remain fenced until authoritative recovery.
- A global published epoch does not authorize a locally stale or not-yet-committed Actor.
- After reconstruction, two Actors regain identical parameter fingerprints/versions; an accepted update can eventually be published.

**NOT proven, and not to be claimed:**
- The upgraded wrapper **does** implement an in-memory epoch ticket counter and reject `begin` while test-wrapper tickets remain active. It has NOT integrated ticket acquisition/release around actual `Learner.update`, all `LearnerGroup` call paths, remote gradient jobs, or crash cleanup for abandoned tickets. Hence real native gradient-update quiescence is **not proven**.
- The coordinator is **not crash durable**, its epoch state and authoritative checkpoint are held in memory, and epoch labels/ACKs are not cryptographically authenticated.
- Accessors to the raw `Learner` outside the wrapper bypass its checks, so this is not a complete Ray LearnerGroup API change.
- Checksums compare RLModule weights, not optimizer state history, selector semantics or device memory. State transfer intent/writer identity remain under-specified.
- Ray actor death, node failures, network partitions, dynamic membership, ACK forgery, restored random-number states, genuine vLLM KV cache epochs and arbitrary concurrent training are not covered.
- No comparison of RL return, true optimizer trajectory recovery correctness against strong production checkpoints, GPU speed, or full upstream Buildkite success is available.

## Review and integration gate

The next engineering step is not to rename this a generalized 2PC innovation. Build a native `acquire(epoch)/release(ticket)` train/update wrapper whose authority refuses staging until all in-flight epoch tickets drain; add a real suspended training-step test and fault injection for actor crashes/ack timeouts. Only then propose a narrowly scoped upstream design, **after** maintainers decide if fail-stop restoration belongs in `Learner`, `LearnerGroup` or the checkpoint coordinator.

The existing official [Ray PR #67131](https://github.com/ray-project/ray/pull/67131) remains Open with Buildkite Microcheck failure; this prototype deliberately does not modify it.


## Third independent native CI: epoch lease drain and TOCTOU reduction

[GitHub Actions #39993877260](https://github.com/lindicaphxag-tech/ray/actions/runs/39993877260) completed **SUCCESS** after an intermediate, transparent failure due to a mistaken expected denial label (`BLOCKED` instead of the more precise `AUTHORITY_FENCED`). The new protocol adds `EpochAuthority.acquire(epoch)` and `release(ticket)`, with `begin(e)` rejecting a new epoch if any outstanding ticket remains.

**New precondition actually tested on two real Ray actors:** Actor 0 held a ticket while the controller requested `begin(2)`. The authority threw a controlled `RuntimeError` without fencing/starting publication. After the actor released the ticket, the same update could begin. Output: `TRAINING_LEASE_DRAIN: rejected begin until in-flight ticket released`.

All earlier independent probes still passed: authority-first fencing before local stop, selector setter exception on actor 1, both-actor authoritative epoch-1 rebuild, failure to publish after an intentionally withheld ACK, second rebuild, publication after both ACKs, delayed local commit notification, finally agreement on committed epoch 2 and the blocked default-policy selector.

Measured single-run operation spans: fault staging **9.18 ms**, first authoritative recovery **28.69 ms**, successful final update **17.15 ms**. Warm-run and instrumentation variability mean no reliable speedup or amortized throughput claim.

**Key remaining gap:** the held training lease was exercised by a deterministic actor wrapper, not a concurrent real `Learner.update` gradient step. Correct production integration must atomically acquire/release tickets around *all* gradient and snapshot operations and handle actor crash/lease expiry without releasing still-active operations prematurely. No new Ray upstream production PR was created by this result.

## Fourth native result: actual gradient-step overlap, not merely a held dummy ticket

[GitHub Actions #39997227527](https://github.com/lindicaphxag-tech/ray/actions/runs/39997227527) finished **SUCCESS**, using installed `ray[rllib]==2.59.0`, CPU PyTorch, native Ray actors, and the unchanged reviewed production `Learner.set_state` source AST. The source-tree CartPole offline fixture was copied into the installed wheel because PyPI does not ship that test JSON.

Actor 0 acquired a central epoch-1 ticket **inside a wrapper method executing the real native `Learner.update(batch=...)`**, not by calling an empty lease probe. The test intentionally paused for one second **inside the native `before_gradient_based_update` hook** after signaling the authority that gradient update execution had entered. During this hook, the coordinator attempted `begin(2)` and received the expected `in-flight training lease prevents new epoch` error. The actual 200-row CartPole update then completed and changed native policy parameters by max absolute **0.0010000020265579224**. The ticket was released in `finally`. The actor was rebuilt from the last authoritative epoch-1 state to avoid contaminating subsequent cross-actor assertions.

The same CI reran the earlier failure matrix: authority-first gate, independent dummy lease drain, injected selector exception in actor 1, authoritative rollback, deliberately missing one stage ACK, both-actor rebuild, and a delayed local commit acknowledgment. Both finished at epoch 2 with matching model hashes and appropriate selector-based blocking.

**What was and wasn't demonstrated:**
- A real gradient update can be in flight while a coordinator concurrently requests a checkpoint epoch change; the experimental central lease denies that change until completion.
- The controlled one-second pause was deliberately injected into a native pre-update hook to guarantee the schedule. It is *not* proof of multi-second GPU kernels or other uninstrumented training paths.
- The wrapper obtains and releases a ticket around this specific real `Learner.update`. It is **not** integrated into all core Ray LearnerGroup calls, inference/EnvRunner operations, native process crash recovery or dynamically spawned actors.
- The measured policy parameter delta is proof of a gradient update, not an improvement in return/reward.
- This single CI does not estimate throughput or lease RPC overhead under load.

**Causal value relative to the previous experiment:** the prior `hold_training_lease()` method only asserted the authority's ticket counter. The current test independently proves the guarded interval contains a real native gradient update with a real batch, and that a concurrent authority request cannot start an in-place checkpoint install during it. This is a stronger correctness experiment, though not a general distributed linearizability theorem.

**Next acceptance requirement:** instrument real parallel `LearnerGroup` / `EnvRunner` pathways, quantify per-step lease overhead versus asynchronous/no-lease and full-barrier baselines, and test actor loss or lease leaks under bounded timeout/recovery before proposing a production API to maintainers.
