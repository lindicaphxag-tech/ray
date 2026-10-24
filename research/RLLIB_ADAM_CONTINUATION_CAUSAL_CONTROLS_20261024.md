# Recovery-trajectory separation: Ray vs independent native PyTorch DDP

**Date:** 2026-10-24, single-machine CPU. **Status:** investigation; no upstream production bug, optimizer implementation change, breakthrough reward result, or official merge is claimed.

## New causal control 1: Ray DDP fresh reconstruction without crashing

[Successful Ray 2.59.0 no-crash DDP reconstruction CI #40012280760](https://github.com/lindicaphxag-tech/ray/actions/runs/40012280760): a warmed two-Learner `LearnerGroup` checkpoint was copied into a *new* two-Learner DDP group without killing any existing process.

- Restored native model max absolute error: **0**
- Restored optimizer max absolute error: **0**
- Reported optimizer parameter-group configuration: equal
- After one subsequent distributed gradient step, new-group vs uninterrupted group model max absolute error: **0.00017115473747253418**
- Optimizer state max absolute error: **0.0034577129408717155**
- New-group cross-peer model max absolute mismatch: **0.00034783780574798584**

This result demonstrates that **actor death is not necessary** for the trajectory discrepancy in this particular Ray test harness. It does **not** identify the cause (it may include custom RLlib test hooks and CPU runtime effects) or prove a Ray product bug.

[Earlier invasive optimizer-step replay CI #40011812451](https://github.com/lindicaphxag-tech/ray/actions/runs/40011812451): cloned Adam and original optimizer matched on healthy Learners, but not on rebuilt Learners. A second [hook/alias audit #40012280649](https://github.com/lindicaphxag-tech/ray/actions/runs/40012280649) produced rank-dependent mismatches. This makes **the instrumentation itself a credible confounder**. These are intentionally red *trajectory* tests, not production fix evidence.

## New causal control 2: fully independent native PyTorch CPU DDP

To separate intrinsic DDP/checkpoint behavior from the RLlib harness, a **separate local native two-process PyTorch 2.10.0+cpu Gloo DDP experiment**, without any Ray or optimizer/gradient monkeypatching, executed:

- 4 seeds: 7, 19, 31, 43; each seed runs three entirely new 2-rank Gloo process groups;
- same deterministic per-rank CartPole-like synthetic regression batch, 2-layer tanh MLP, real DDP gradient reduction and Adam optimizer;
- group A: warm update, checkpoint model+optimizer, continue one update;
- group B: restore full warmed checkpoint in a fresh new DDP process group, continue identical update;
- group C (negative control): restore **only model weights**, start with fresh Adam, continue identical update;
- compare model, optimizer, inter-rank equality after step (exact float tensors).

**Observed results (absolute max across tensors)**

| Seed | Full checkpoint next model mismatch | Full checkpoint next Adam mismatch | Weight-only next model mismatch | Full checkpoint DDP peer mismatch |
|---:|---:|---:|---:|---:|
| 7 | 0 | 0 | 0.0000102892518044 | 0 |
| 19 | 0 | 0 | 0.0014520585537 | 0 |
| 31 | 0 | 0 | 0.0000760853290558 | 0 |
| 43 | 0 | 0 | 0.0000043809413910 | 0 |

Three seeds ran in the first local container job; seed 43 ran in a separate job after the first command reached its 120-second execution limit. The **four seed results were all independently observed**, but this is NOT one uninterrupted 4-seed CI run. The source was preserved as a standalone `ddp_adam_resume_isolation.py` user artifact rather than represented as GitHub CI. CPU PyTorch version may not match the native Ray workflow's installed version.

**What these controls establish:** under this tested basic DDP setup, independent process-group reconstruction and full Adam checkpoint restoration can preserve the immediate next update *exactly*, and failing to restore optimizer state can demonstrably change the next update. Thus DDP and Adam are not intrinsically incapable of exact next-step continuation. The Ray mismatch requires additional isolation from instrumentation, module/config, distributed state or optimizer loader behavior.

**What they do not establish:** They do not establish that the observed Ray 2.59 failure is caused by RLlib, or that it persists in an uninstrumented Learner or across installations, data distributions, GPU backends, random seeds, longer trajectories, or production training.

## Scientific next discriminator

A submitted [fork-only noninvasive native Ray four-seed negative-control branch](https://github.com/lindicaphxag-tech/ray/pull/13) now calls the *actual unmodified* `Learner.get_state/set_state/update` on a warmed local learner and an independently constructed restored learner, with no optimizer/gradient instrumentation. **As of this document GitHub has not reported an Actions run for that branch; do not claim a Ray result.** If local uninstrumented Learner is exactly reproducible, reintroduce distributed LearnerGroup and then simulated versus actual actor death **one factor at a time**. If it diverges, inspect state identity, RNG, step scheduling, tensor aliasing and optimizer state binding and design the smallest independent reproduction before proposing a patch.

## Candidate criterion for a representative contribution

A mere crash recovery success is insufficient. Require a proper `RecoveryTrajectoryCertificate` over `(weights, optimizer moments, parameter identities, scheduler/RNG, training selector, topology epoch)`, with:
1. identical source/target states before a step;
2. a fresh independent checkpoint restore implementation (and **matched no-crash control**);
3. exact next-step or justified numerical-tolerance comparison on fixed, explicitly identified inputs;
4. real Ray Actor death, two Learners, DDP and independent settings;
5. production code change restoring the invariant, followed by up-to-date native CI and upstream maintainer acceptance.

Until (1)-(5) hold, this is an experimental finding and a rigorous experimental design, not a novel accepted algorithm or official merge.
