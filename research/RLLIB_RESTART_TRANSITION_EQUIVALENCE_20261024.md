# RLlib DDP restart transition equivalence: fault-independent native counterfactual

**Date:** 2026-10-24. **Status:** reproducible negative research evidence; not a merged upstream fix, an established new Ray regression, or a novel general theorem.

## Contract stronger than checkpoint equality

For a training state `S=(W,O,T,C)` (parameters, optimizer, trainability selector, remaining execution context), per-rank batch `B`, and training transition `U`:

- Weak restore criterion: `project(S_rebuilt) == project(S_reference)`, commonly checking only W and some optimizer buffers.
- Required continuation criterion: `U(S_rebuilt,B)` is equivalent to `U(S_reference,B)` for the declared determinism tolerance and execution backend.
- The implication **weak equality => transition equivalence** is unsupported if relevant execution context, parameter/optimizer bindings or update semantics are not captured.

This is a known concept (observational equivalence/bisimulation), not a new theorem. The research contribution candidate is finding a **specific observable native Ray/RLlib case where the weak criterion passes yet the stronger one fails**, then isolating a correct and efficient repair.

## Independent native counterfactual: NO actor failure required

**[Successful diagnostic Actions #40019181402](https://github.com/lindicaphxag-tech/ray/actions/runs/40019181402)** uses installed Ray 2.59.0, 2 real CPU/PyTorch DDP Learners, and a genuine 200-row CartPole batch.

1. Run actual gradient to populate nonzero Adam state and take a deep-copied full LearnerGroup checkpoint.
2. Keep healthy group and execute one more real DDP gradient step: the reference continuation.
3. **Without killing any actor**, shut down healthy group, create a new 2-Learner group, `set_state` using exactly that warm checkpoint and execute the same batch.
4. Compare recoverable weights, optimizer numeric buffers (12 measured), param group settings, and next-step state.

Single CI observations:
- Rebuilt model-state max error at admission: **0**.
- Rebuilt Adam numeric state max error at admission: **0**.
- Restored optimizer parameter-group configuration: identical.
- Reference versus new group after one native update: **0.00027257204** max weight difference; **0.00349208340** max optimizer-state difference.
- Within newly rebuilt group: **0.00028733909** max cross-Learner parameter difference.
- Healthy rebuild + restore wall time: **3839.956 ms**. Single CPU runner; not a throughput benchmark.

This rules out physical actor death as a necessary cause. The CI is **green because it is a diagnostic that requires completion, not because numerical parity succeeded**. The numeric mismatch is recorded explicitly.

[Exact unmodified-training test source](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_native_ddp_no_crash_rebuild_control.py).

## Adversarial test without touching Adam.step

**[Separate CI #40019181381](https://github.com/lindicaphxag-tech/ray/actions/runs/40019181381)** follows the physical-actor-loss scenario, but its observer records the full optimizer state/weights/gradient inputs *before* calling the original bound `TorchLearner.apply_gradients`; the production `Adam.step` is never monkeypatched. An independent new Adam is created and stepped **only after the native gradient application has returned**.

- In the two healthy Learners: original-vs-independent one-step Adam replay max errors **0 / 0**.
- In the two rebuilt Learners: original-vs-independent replay max errors **0.00038440153 / 0.00060452521**.
- Each rank's actual training input and raw-gradient SHA-256 agree between healthy/rebuilt; gradient all-reduce agreement is checked.
- The rebuilt native learners remain synchronized within **0.00041446090** parameter difference after the next step (not exact).
- This CI correctly fails its strict one-step trajectory assertion.

[Noninvasive observer source](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_native_noninvasive_adam_replay.py).

## Other causes already tested and ruled out *within the observed scope*

- [Adam input and output boundary audit #40011812451](https://github.com/lindicaphxag-tech/ray/actions/runs/40011812451) compared exactly captured native optimizer gradients and independent replay, but that earlier experiment monkeypatched the native step.
- [Native tensor layout/storage audit #40018685441](https://github.com/lindicaphxag-tech/ray/actions/runs/40018685441) found no cross-optimizer storage overlap, no parameter-versus-moment storage overlap, identical clone tensor shapes/dtypes/devices/strides, and no relevant instance optimizer-step hooks. The divergence remained. **A negative result about observed pointers is not proof that all external tensor aliases are absent.**
- Source-exact prior audits report equal checkpoint W/12 Adam numeric buffers, optimizer per-parameter name binding, input batch, pre-apply gradients, and optimizer hyperparameters.

## Scientific boundaries / integration decision

**Unresolved:** why the production native and independent PyTorch Adam updates sometimes differ despite the inspected equality conditions; whether parameter-group internal bookkeeping, torch execution context, DDP process state, snapshot timing, or extra unrecorded tensors are responsible; cross-version reproducibility; generality beyond this testing Learner.

Do not assert a PyTorch bug, a Ray upstream production regression, general nondeterminism, optimizer state aliasing or a repaired protocol on this evidence alone. Nor should one force a green result by widening `atol` or deleting strict continuation tests.

**Next discriminating tests** (pre-registered):
1. Pure Torch Adam warm restore with same named parameters/gradients and full optimizer object snapshots, no Ray or DDP; a positive control expected to reproduce the transition exactly.
2. Single native RLlib Learner rebuilt from warm checkpoint with same batch and no DDP; if it fails, reduce problem before DDP analysis.
3. Two native RLlib Learners without DDP using matched per-rank batches; compare a zero-crash reconstruct and `torch.optim.Adam` original state without any tracing, across fresh seeds and CPU Torch versions.
4. If the mismatch survives instrumentation-free positive controls, construct minimal failing upstream test that checks **the next model/optimizer transition**, not only checkpoint tensor equality, and propose the narrowest targeted repair.

**Prior successful results remain distinct:** [real physical actor loss and full group rebuilding with resumed gradients](https://github.com/lindicaphxag-tech/ray/actions/runs/40005461182) showed operational recovery, not numerical trajectory equivalence; [native async queue drain](https://github.com/lindicaphxag-tech/ray/actions/runs/40002895154) showed gating and no new dispatch, not this stronger property.

Fork [research Draft #11](https://github.com/lindicaphxag-tech/ray/pull/11) stays separate from official still-open [Ray #67131](https://github.com/ray-project/ray/pull/67131) and [#67329](https://github.com/ray-project/ray/pull/67329). No new external merge has been claimed.
