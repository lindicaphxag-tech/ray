# Non-invasive Adam continuation: falsification protocol (2026-10-24)

**Status: experimental plan, not an executed test or upstream bug claim.** This document is deliberately kept in the author's fork research branch, separate from ray-project/ray#67131 and #67329.

## Why the current evidence is insufficient

Native two-LearnerGroup tests established a real post-checkpoint one-step discrepancy under a warmed Adam state, including a no-actor-death control. The later diagnostic wrapped the *live* `Adam.step()` and performed an independent step inside that wrapper. The recovered ranks' replay discrepancies differed between executions. Consequently, a non-invasive, pre-captured oracle is needed before blaming Ray checkpoint serialization, tensor aliasing, PyTorch, or DDP.

## Pre-registered tests (in causal order)

1. **No-rebuild negative control:** train two fully independent healthy groups from synchronized snapshots and verify equal per-rank input, pre-step parameter, gradient and optimizer states; compare next-step outputs, without restoring or killing either group. Failure invalidates recovery-specific attribution.
2. **Standalone CPU Adam oracle:** *outside* Ray and without monkeypatching production optimizer methods, clone model tensors, `grad`, all optimizer parameter groups, per-parameter state, and tensor metadata. Execute two freshly created Adam optimizers on independent deep clones. Require self-replay equality before comparing any native result.
3. **Frozen pre-step capture:** with an explicit barrier **before** calling the original `Adam.step()`, copy all relevant state to a file/artifact and release the native step without executing another optimizer in the same process. Replay the captured tensors in a **separate process** only after native execution finishes. Captures must not modify tensor storage, parameter bindings or `grad`; confirm capture hashes are stable.
4. **Object and layout audit:** record `id(Parameter)` membership in each optimizer group, `optimizer.state` key-to-Parameter identity, storage pointers and version counters (diagnostic only), tensor dtype/device/shape/stride/contiguity, `step` scalar type and dtype, `foreach`, `fused`, `capturable`, `differentiable`, `maximize`, `amsgrad`, LR scheduler state, grad scaler state, and whether any gradient is `None`. Record PyTorch and Ray exact versions and CPU threading/BLAS configuration.
5. **Kernel-mode cross-control:** explicitly fix Adam `foreach=False,fused=False` in *both* groups, then repeat with `foreach=True` where supported. Do not compare configurations or infer a bug from expected floating-point differences between kernels.
6. **DDP causality:** first use a single Learner with warm Adam, then two Learners with a fixed per-rank batch and synchronized step barrier. Verify actual optimizer-read gradients, not merely all-reduce outputs. Keep equal-world-size and identical rank-to-data mapping.
7. **RNG and timing controls:** save/restore Python, NumPy and torch RNG states at the same logical boundary; test deterministic CPU settings, same-process repeated continuation, independent seeds, and rank-order swaps. Repeat each arm independently and summarize both failures and successes.

## Acceptance and interpretation

For matched pre-step state `S=(W,O,G,C)`, where `C` includes optimizer configuration and relevant execution metadata, require (i) independent-oracle self-consistency; (ii) original native step vs offline oracle agreement at an explicitly predeclared absolute/relative tolerance, using exact bytes when mathematically expected; and (iii) uninterrupted vs restored continuation agreement across ranks. Report max absolute and relative error plus per-tensor mismatch hashes. Never quietly raise tolerances post hoc.

- **If two offline identical clones diverge:** the oracle or runtime determinism premise is broken; do not call this a Ray bug.
- **If offline clones agree but healthy native groups disagree:** optimize/test runtime or audit first; recovery causality remains unproven.
- **If healthy groups agree and restored native differs with equal actual Adam inputs:** investigate an unrecorded optimizer execution dependency, verify on fresh upstream head and in standalone PyTorch, then prepare the smallest red/green upstream reproducer.
- **Only after a causal patch restores the full next-step parity:** consider a separate feature proposal for efficient epoch-fenced checkpoint admission and compare its cost against existing whole-group rebuild.

**No merge-readiness claim.** A successful diagnostic workflow means the probe executed; it is not proof of trajectory-equivalent recovery. This file introduces no Ray production patch and no externally merged PR.
