# RLlib three-component causal state admission — October 23, 2026

**Research evidence, not a merged upstream contribution or a new RL optimizer.**

## Finding and actual training effect

Learner.get_state exports the trainability selector should_module_be_updated
together with the weights sequence and (optionally) module/optimizer state.
Learner.should_module_be_updated(module_id) reads this selector to decide
which policy participates in the next update.

The earlier fork candidate prevented stale model/optimizer partial restore
but still accepted the stale snapshot's training selector unconditionally.
A rejected v9 or equal-version foreign v10 full snapshot could therefore
preserve the live model and Adam states while switching the next training
target from main to opponent.

The new candidate uses a single module_update_accepted predicate computed
BEFORE mutating the installed version. All attached optimizer/selector
components follow that admission decision. Standalone optimizer or selector
updates keep their independent API behavior.

## Formal transition contract

Let S = (v, theta, m, p) be the live version, module parameters, optimizer
history and trainability selection. A received packet is a subset of
R = (v', theta', m', p'). Let

    A = has(theta') and (v' == 0 or v' > v)

and define component transitions by

    theta+ = theta' if A else theta
    m+ = m' if has(m') and (not has(theta') or A) else m
    p+ = p' if has(p') and (not has(theta') or A) else p
    v+ = v+1 if A and v'==0 else v' if A else v

Case split: for no model, independent optimizer/selector updates do not
publish a new model version; for rejected model, its attached optimizer
and selector cannot change; for accepted model, all supplied associated
components are applied and its new version is published.

This is **successful-call admission consistency**, not ACID rollback on
exceptions, not a guarantee of consistency during concurrent component
writes, and not proof of globally unique state identity from scalar v.

## Real native test evidence

The red control is the PREVIOUS, otherwise passing two-component fork patch
at commit c31c743e3faf4d304e96e0da158ceee34d3f1a54.

Four methods from the actual Ray TestLearner native test class were executed
against the real Ray 2.59.0 Learner environment with only the production
set_state function replaced from source. The three existing tests passed
in both variants; the new trainability-selector contract failed ONLY on
the old variant. The new candidate passed all four and checked the actual
should_module_be_updated(DEFAULT_MODULE_ID) training decision.

[Native source red/green run](https://github.com/lindicaphxag-tech/ray/actions/runs/39915547493)

## Bounded packet-state verification

A source-driven Python checker exhaustively ran every sequential trace of
one, two or three packets from 52 types (full, paired, model-only,
model+selector, optimizer-only, selector-only, metadata-only), rates
0/1/2/3, and independent sources A/B:

    52 + 52^2 + 52^3 = 143,364 traces per candidate

Previous two-component version: 43,784 traces violating selector provenance.
New three-component version: 0 violations of all specified finite-model
state acceptance and publication properties.

[Source-driven bounded model check](https://github.com/lindicaphxag-tech/ray/actions/runs/39914974923)

This does NOT exhaust arbitrary real multithreading, actor failures,
intra-method interruption, exception rollback or network delivery behavior.

## Small maintainer-review diff

[Two-file clean diff](https://github.com/lindicaphxag-tech/ray/compare/fix/rllib-version-unversioned-set-state-20261018...review/rllib-three-component-causal-state-20261023)

It changes production learner.py and native test_learner.py only.
The source-extraction scripts and CI workflow are NOT part of the clean
review branch. Relative to the already-open #67131 head, the clean
candidate changes approximately 25 added/14 removed production lines
and 105 added/1 removed native test lines. No new public API or class.

Open semantics decision: should an equal-version authoritative checkpoint
force restore ALL matching components, or be a no-op? The candidate keeps
the existing strict-newer module admission and rejects an equal-version
bundle as a whole. A force-restore operation would need a separately
reviewed contract.

[Ray official issue #67292](https://github.com/ray-project/ray/issues/67292)
[Existing Ray PR #67131](https://github.com/ray-project/ray/pull/67131)

## Originality and next gate

This is a project-specific correctness advance using established
state-consistency ideas, not evidence of a novel generic RL algorithm.
For a meaningful L8+ research advance, an identity-aware publication
protocol must cover real asynchronous Learner/EnvRunner traffic, recover
from cancellations/failures, avoid spurious drops, and beat strong existing
recovery strategies. The first merge gate is maintainer acceptance and
a passing official Buildkite Microcheck; neither has occurred yet.
