# Warm Adam continuation is NOT certified by identical checkpoint fields

**Observed date:** 2026-10-24 (Asia/Singapore) | **Platform:** native Ray 2.59.0, PyTorch CPU, 2 real DDP Learners on CartPole | **Status:** a reproducible **negative result**, root cause **NOT established**. This is a fork-only diagnostic, not an accepted upstream RLlib defect/fix.

## Research question

The previously successful [real actor crash -> whole-group rebuild -> one real DDP gradient](https://github.com/lindicaphxag-tech/ray/actions/runs/40005461182) checked that the rebuilt group's parameters matched the old checkpoint and subsequent training produced nonzero updates. That is weaker than **trajectory equivalence** against an uninterrupted warm-optimizer control.

For a committed checkpoint \(S=(W,O,T,e)\), does reinstantiating Learners from \(S\) and applying the same distributed batch reproduce the uninterrupted reference optimizer update, not merely match weights immediately after loading?

**Experimental arms:**
1. Run a real DDP gradient on a 2-Learner native group to create **nonempty Adam moments** and save a deep-copied warm checkpoint.
2. Healthy reference: execute one more DDP gradient, capture all per-worker model weights, optimizer moments and gradient evidence.
3. Crash/rebuild: physically terminate a Ray Learner, independently confirm physical death, fail a fresh membership challenge, discard damaged group, create a fresh 2-worker DDP group, load exactly that warm checkpoint, and execute the same batch once.
4. Compare next-step weights and optimizer buffers, not just initial restore; verify the exact per-worker objective batch and gradient hashes, Adam parameter binding and optimizer configuration.

Unlike an unsafe **in-place backward restore to a still-live Learner**, the recovery arm uses *new objects/processes*. The original attempt to rewind old positive version state inside an active Learner **failed** the weight-equality assertion [CI #40007462225](https://github.com/lindicaphxag-tech/ray/actions/runs/40007462225); this does not establish a bug in a documented checkpoint API, since positive weight version admission can reject old weights and rollback semantics need to be specified.

## Strong negative evidence, independent CPU reruns

- [CI #40007815235](https://github.com/lindicaphxag-tech/ray/actions/runs/40007815235): warm checkpoint contained **12 numerical Adam buffers**, rebuilt worker state matched the checkpoint (max error **0**), but the recovered DDP learners were not trajectory-equivalent under a strict 1e-5 parameter tolerance.
- [CI #40008218385](https://github.com/lindicaphxag-tech/ray/actions/runs/40008218385): healthy warm continuation peers had **zero parameter/optimizer discrepancy**, and the newly rebuilt peers had **zero optimizer-state discrepancy before training**. However next-step model difference between recovered and uninterrupted control was **0.0001707226**, and a recovered-peer difference was observed in this run (**0.0003992915**).
- [CI #40008624870](https://github.com/lindicaphxag-tech/ray/actions/runs/40008624870): actual data arrays reaching the per-module loss functions had the same per-worker SHA-256 digests in both arms; a recovered next-step model error remained **0.0005154312**.
- [CI #40009032384](https://github.com/lindicaphxag-tech/ray/actions/runs/40009032384): all four actual pre-postprocess model-parameter gradients had **identical SHA-256 hashes** in both arms and across peers, while next-step model differences persisted.
- [CI #40009391904](https://github.com/lindicaphxag-tech/ray/actions/runs/40009391904): actual Adam parameter group layout, order of four named model parameters and hyperparameters (including lr=0.001, betas=(0.9,0.999), eps=1e-8) matched between the two arms.
- [CI #40009798959](https://github.com/lindicaphxag-tech/ray/actions/runs/40009798959): each Adam exp_avg, exp_avg_sq and step was verified *bound to the same named live PyTorch Parameter* via its own SHA-256, not merely stored in the same numbered state_dict slot. The name-to-moment associations matched between the arms.
- [CI #40010200512](https://github.com/lindicaphxag-tech/ray/actions/runs/40010200512): immediately before the actual `apply_gradients -> optimizer.step`, the **postprocessed** per-parameter gradient SHA-256 hashes, l1 magnitudes, optimizer group config and parameter names also matched. Still, this run's next-step recovered-vs-reference model maximum absolute error was **0.00030213594**, with recovered next-step Adam-state difference **0.0040186755**.

The strict continuation-equivalence assertion intentionally **failed** in these diagnostic runs. **Do not misrepresent any of them as all-green validation of recovered Adam trajectories**, nor as proof of a high-level algorithm improvement. The older all-green CI checks only initial state consistency and subsequent real gradient execution, a lower bar.

## What the evidence supports — and does not support

**Supported:** A live warm Adam checkpoint cannot yet be declared sufficient for deterministic *one-step replay* of the instrumented multi-Learner native training path. Model tensors, serialized optimizer buffers, actual optimizer-to-parameter moment bindings, model input data and actual postprocessed gradients have all matched at the documented probes in repeated diagnostics, while the next weights differ above strict tolerance.

**NOT supported:** The evidence does not yet show this is an upstream Ray defect or that the physical actor failure *causes* the divergence. A critical next negative control is a **fresh two-LearnerGroup rebuild from the same checkpoint with NO actor death**, matched to a healthy continuation. An exact-run same-process-repeatability control and implementation-independent torch.Adam step oracle are also missing. The next-step difference could come from a hidden optimizer/runtime execution state, optimizer scheduling outside serialized fields, instrumentation timing, incomplete test harness controls, or nondeterministic runtime behavior. The current evidence does not distinguish them.

**Do not misstate** identical module and optimizer-state hashes as guaranteeing entire execution identity: PyTorch backend and LR scheduling state, random streams and framework internals require their own controlled checks. The current BC test's forward path has no obvious dropout, but this alone does not prove deterministic distributed training.

## Proposed higher-tier scientific gate

An eventual transaction-safe checkpoint protocol must specify *two separate* correctness contracts:

**(A) Publication safety:** no incomplete or stale components are made available to the learner/EnvRunner after a failed or missing participant ACK.

**(B) Training continuation:** for a declared deterministic training mode and exactly matched per-worker data/optimizer execution conditions, replay of the accepted checkpoint yields the expected next optimizer state and model update to a predeclared tolerance.

Existing experiments support restricted (A) with a userland coordinator and native tests. The new negative result **blocks claiming (B)**.

**Priorities:** (i) fresh rebuild without crash control, (ii) same-environment replay and independent PyTorch Adam numerical oracle, (iii) no hidden grad scaler/learning rate/step/buffer remapping, (iv) actual multi-worker optimizer-trajectory parity, (v) only then compare optimizations over the measured 4.28-second whole-group restart baseline. Avoid pushing a speculative production patch into official #67131 or #67329.

**Current external PR status:** [Ray #67131](https://github.com/ray-project/ray/pull/67131) and [#67329](https://github.com/ray-project/ray/pull/67329) remain Open; this diagnostic is in [fork Draft #11](https://github.com/lindicaphxag-tech/ray/pull/11), not an official Merge.

## Reproduction

[Source instrumenting actual Ray LearnerGroup workers](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_native_two_learner_ddp_crash_health.py)

[Exact native CI definition](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/.github/workflows/rllib_native_two_learner_ddp_crash_health.yml)


## Decisive no-actor-crash control — completed successfully

[Native Ray 2.59.0 two-Learner DDP GitHub Actions #40011006462](https://github.com/lindicaphxag-tech/ray/actions/runs/40011006462) completed a standalone **no-kill** experiment. It built a healthy 2-LearnerGroup, executed a genuine update to warm Adam, saved the complete checkpoint, executed another healthy DDP update as the reference, **shut down the healthy group without physically killing an Actor**, created a brand-new group with the same configuration, restored the warm checkpoint and trained the same supplied batch object once more.

Observed values (one CPU CI run):

| Metric | Value |
|---|---:|
| Saved numerical optimizer buffers | 12 |
| New group initial model tensor max error vs checkpoint | **0** |
| New group initial numerical Adam-buffer max error vs checkpoint | **0** |
| Optimizer param-group configuration match | true |
| Next-step model max difference vs uninterrupted healthy group | **0.00037486106157302856** |
| Next-step optimizer max difference vs uninterrupted healthy group | **0.00190426642075181** |
| Next-step mismatch between two workers in recreated group | **0.00037267059087753296** |
| Fresh 2-worker group creation + warm restore span | **4,468.65 ms** |

**Result:** `NO_CRASH_RECONSTRUCTION_TRAJECTORY_NON_EQUIVALENCE_OBSERVED`; `NO_CRASH_CONTROL_COMPLETED`; workflow **success** signifies successful execution of the *diagnostic*, NOT successful trajectory parity.

**Causal attribution improved:** the observed warm-checkpoint next-update divergence **does not require a physical Ray Actor crash**. The independent crash/rebuild diagnostic is compatible with a more general fresh-LearnerGroup continuation boundary or a nondeterministic aspect of the test/runtime. Calling the effect 'caused by crash recovery' is therefore unsupported.

**Still not proven:** This particular no-kill script reuses the same input MultiAgentBatch object, but does **not itself** hash each per-Learner sharded objective batch and gradient. Separate physically killed-worker diagnostics did show equal per-rank SHA256 for actual inputs, backward gradients and postprocessed gradient tensors. A definitive production root-cause claim still needs no-kill per-rank batch/gradient audit and no-rebuild repeated healthy controls, ideally against the current Ray upstream commit, to rule out harness or runtime details.

**Reproduction code:** [no-crash control](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_native_ddp_no_crash_rebuild_control.py), [workflow](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/.github/workflows/rllib_native_ddp_no_crash_rebuild.yml).

### Implication for the flagship research direction

Before pursuing *low-cost partial Actor repair*, define and verify a **sufficient distributed training checkpoint** including model weights, optimizer buffers, parameter mapping, selected trainability rules, declared input/shard provenance and any runtime/scheduler state that materially changes the next optimizer step. An optimized repair that retains inaccurate next-step dynamics has not met the algorithmic standard, regardless of its restart-time speedup.

The current evidence falsifies simple claims of warm-Adam one-step parity under the tested setup; it does not prove a generally new checkpoint algorithm, nor that Ray lacks a supported way to restore a valid distributed training configuration.

**Next minimum experiment:** actual per-worker postprocessed gradient hashes in the no-kill control, followed by a same-process/no-rebuild replay baseline that respects weight-sequence admission semantics; investigate optimizer update internals, scheduler state and DDP collective reinitialization. If reproducible on a supported current upstream branch, submit the smallest causal reproducer to maintainers before proposing implementation.
