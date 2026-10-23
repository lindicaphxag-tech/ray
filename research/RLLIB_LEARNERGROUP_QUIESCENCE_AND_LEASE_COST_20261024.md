# RLlib async checkpoint quiescence and amortized epoch tickets

**Status:** Independent research on real Ray 2.59.0 native CPU Learner and LearnerGroup. Neither an official Ray patch nor evidence of improved RL reward. The existing issue [ray#67292](https://github.com/ray-project/ray/issues/67292) remains open and any future production change needs API/maintainer feedback.

## Precise integration obstacle discovered in upstream Ray

In `python/ray/rllib/core/learner/learner_group.py`, the remote `LearnerGroup.update(async_update=True)` calls `self._worker_manager.fetch_ready_async_reqs(timeout_seconds=0.0)` to collect previous completed results, then **immediately calls `foreach_actor_async('update', ...)`** to submit new work. In contrast `LearnerGroup.set_state` calls its local Learner or `foreach_learner(lambda l: l.set_state(...))` to install a checkpoint. There is **no shared transaction/barrier guarantee** for arbitrary concurrent update submission and multi-component state installation.

The `FaultTolerantActorManager` already exposes the lower-level facilities necessary for a scoped, exclusive coordinator:

- `num_outstanding_async_reqs()` tracks queued actor update requests;
- `fetch_ready_async_reqs(timeout_seconds=...)` collects *without submitting any new update*;
- by contrast, using `LearnerGroup.update(async_update=True)` to 'drain' creates new work, so it is not a quiescence primitive.

A sound userland barrier needs: (a) stop all new update submissions through one exclusive entry, (b) await manager outstanding count == 0 and inspect all errors, (c) only then apply full versioned checkpoint state, and (d) atomically re-open admission. This is **not** a guarantee if other code can call the raw LearnerGroup concurrently, if external actors bypass this coordinator, or under actor failures that clear manager counters without completing work.

## Lease batching: sampled wall-clock cost

[Successful native Ray CPU/PyTorch GitHub Actions #40002166087](https://github.com/lindicaphxag-tech/ray/actions/runs/40002166087). A single native Actor executed genuine `Learner.update` on 200-row CartPole data; each arm reset from exactly the same authoritative weights and replayed the same batch, with four rotated-order blocks x 16 native gradient updates. Each arm had the **same actual final parameter change (0.0165899843)** in all four blocks.

Median time per 16 updates (ms):

| Mode | Median ms | Ratio to raw | Ratio to per-update lease |
|---|---:|---:|---:|
| No remote lease (unsafe baseline) | 28.643 | 1.000 | 0.405 |
| Acquire/release via remote coordinator for *every* update | 70.729 | 2.469 | 1.000 |
| Acquire/release one remote ticket per 8 updates | 34.722 | 1.212 | 0.491 |

Four raw duration lists (ms):
- no lease: [32.3810, 29.7910, 27.4945, 27.1100]
- each update: [70.1690, 80.1896, 71.2895, 65.2938]
- amortized (K=8): [34.4382, 35.5793, 32.3193, 35.0063]

**Causal interpretation:** the experiment separates per-gradient *remote actor RPC tax* from actual PyTorch model updates under the tested workload. It does not quantify GPU synchronization, environment rollout cost, Ray's real DDP group overhead, or reward. Median of four blocks is not a confidence interval.

## Correctness vs latency theorem (restricted model)

Suppose all training steps enter only through an epoch authority. A `K`-update ticket is acquired atomically with the authority's `begin(e+1)` operation, and `begin` rejects while any ticket is active. Assume step execution time is at most `Tmax`, actor code releases tickets on completion, and the actor neither crashes nor holds indefinite RPCs.

- **Safety:** no accepted new epoch can be staged during any of the `K` updates holding that ticket, because `begin` sees nonempty outstanding tickets. The state cannot be replaced concurrently *through that coordinator*.
- **Worst-case waiting:** if a ticket spans at most K steps and all ticket-holder computation is bounded, an arriving state update may have to wait **up to approximately K*Tmax + protocol overhead** for the last acquisition to release. Without finite `Tmax` or a sound lease-expiration/actor-failure policy, there is **no finite upper-bound guarantee**.
- **Cost model:** if acquiring and releasing a remote ticket costs `C` combined, distributed coordination cost per update decreases approximately from `C` to `C/K`, at the expense of delaying checkpoint admission. K cannot be selected from performance alone.
- **Deadline guidance:** under an explicit maximum state-update stall `D`, a fixed K must respect `K*Tmax + protocol_bound <= D` or release the ticket before the deadline. This is a standard batching/latency tradeoff, not an independent novel theorem.

**Security/correctness limitations:** authority actor death, duplicate or forged ACK, out-of-band LearnerGroup methods, dynamic group membership, GPU kernels, snapshot integrity, training rollout source metadata and OPD KV cache provenance remain outside the experiment.

## Immediate upstream-facing decision

Do not add remote per-step leases directly to official Ray PR #67131. It would increase the numerical/engineering cost of short CPU gradient steps by ~147% in the tested workload and could introduce new failure modes. A scoped group-level quiescence barrier or bounded amortized epoch tickets are more plausible. Benchmark full LearnerGroup async queue management and inspect maintainer preferences before proposing production interfaces.

## Evidence and code

- [Lease A/B native benchmark script](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_native_lease_amortization_benchmark.py)
- [Native LearnerGroup async quiescence experiment](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_native_group_async_drain.py)
- [Previous real native concurrent gradient ticket experiment](https://github.com/lindicaphxag-tech/ray/actions/runs/39997227527)
- [Underlying snapshot fault/ACK loss protocol](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/research/RLLIB_EPOCH_FENCED_TWO_ACTOR_20261024.md)
