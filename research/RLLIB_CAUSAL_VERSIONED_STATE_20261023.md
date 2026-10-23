# RLlib causal version and optimizer-lineage contract (research note)

Date: 2026-10-23. This records a production-source correctness investigation,
not an upstream-merged fix or a newly invented RL optimization algorithm.

## Model and invariants

Represent a Learner as S=(theta, m, v), where theta contains neural weights,
m contains optimizer moments and counters, and v is its weight update sequence.
Incoming state R=(theta' optional, m' optional, v') can contain partial
component selections. Ray Learner.get_state emits v even when only the
optimizer is selected, so the arrival of v' alone must not mean that weights
theta' were actually installed.

Invariant I1 (weight causality): an external state operation cannot advance
v unless it actually installs module weights. A local training update remains
authorized to increment v itself.

Invariant I2 (paired admission): for a versioned state containing both
weights and optimizer, either admit both components or reject both. This is
only ACCEPTANCE consistency: if one setter raises after the other has
succeeded, this minimal patch does NOT implement rollback.

Invariant I3 (component isolation): optimizer-only restore remains a valid
operation but cannot certify a new model-weight version. Unversioned module
updates continue to increment v.

Invariant I4 (identity limitation): equality of two scalar weight-update
counters is NOT evidence that two independent Learners hold the same
weights or optimizer history.

## Proposed minimal state machine

For an incoming packet containing model weights, accept those weights only
when v'==0 or v'>v, retaining the existing production rule.

For an incoming packet containing both weights and optimizer:
- v'>v: accept both and publish v';
- 0<v'<=v: reject both; keep v;
- v'==0: accept both and increment v once.

For an optimizer-only or metadata-only packet:
- optimizer may be loaded according to the component-only API;
- v must NOT change without an actual model-weight update.

For an incoming module-only packet:
- preserve the current version acceptance semantics;
- publish the version only after installing the model state.

The equal-version rejection is consistent with the existing strict-newer
module guard. If maintainers intend an equal-version checkpoint operation
to forcibly restore an authoritative full snapshot, that is a DIFFERENT
contract and needs a separate explicit override; an optimizer-only partial
restore should not silently implement it.

## Reproducible counterexamples against real source and real Adam

Test 1: delayed paired checkpoint v9 sent to a live v10 Learner.
The original code retains theta(v10) and installs m(v9). A subsequent real
Adam step differs from a consistent v10 reference by L2=0.000322280417.
The patched source keeps both v10; next-step L2 difference is zero.

Test 2: two Learners independently trained for 10 genuine Adam updates,
on different data/targets, have equal v=10 but different theta and m.
The original code retains the local theta but loads the remote m.
The next Adam step differs from a consistent local v10 reference by
L2=0.051059705430. The candidate rejects the entire equal-version
paired state; the measured next-step difference becomes zero.

Test 3: a component-filtered optimizer-only v11 packet arrives at learner
weights v10 before the matching module v11 packet.
The original set_state prematurely announces v11 without receiving weights,
then rejects the true module v11 on arrival. The candidate does not
advance v for an optimizer-only packet, so the later module v11 is admitted.

The CPU source-based test runner actually compiles the production
Learner.set_state method from the fork checkout. Its baseline uses
the unchanged version from the preexisting Ray PR head:
c8ce9ca7c0ba7b2a15e002b139ff38b99895fbb8.
Same seeds, same torch optimizers and identical request sequences are
used for both versions.

Successful red/green GitHub Actions:
https://github.com/lindicaphxag-tech/ray/actions/runs/39903883741

Fork production candidate:
https://github.com/lindicaphxag-tech/ray/compare/fix/rllib-version-unversioned-set-state-20261018...research/rllib-stale-optimizer-transaction-20261023

Official upstream design discussion:
https://github.com/ray-project/ray/issues/67292

## Limits and higher-level research

These tests establish that a version counter can be a false identity
when detached from actual installation events or used across
independently updated learners. They do not prove global transactional
checkpoint safety, actor-failure recovery correctness, or a change in
real RL returns. Native Ray Learner integration and multi-worker
fault injection remain mandatory for stronger claims.

A potential research-level follow-up is a source-authenticated parameter
publication protocol with explicit ACK, writer epoch/identity, optimizer
lineage and independent verification under cancellation/eviction. That
mechanism must be compared with existing distributed checkpoint and
asynchronous RL schemes before originality or L9 claims are justified.

Do not mix this proposed change into Ray #67131 without maintainer
agreement; it is a separate checkpoint semantics question.


## Upgraded evidence: native TestLearner and bounded exhaustive sequencing

2026-10-23. Three methods from the checkout's original `test_learner.py` native
TestLearner class were executed unchanged in Ray 2.59.0 with actual RLModule
state methods: the unmodified production method failed all three, and the
minimal patch passed all three. The two variants used the same native tests.
[Native red/green GitHub Actions](https://github.com/lindicaphxag-tech/ray/actions/runs/39911970229).

A further bounded transition enumeration used 27 types of incoming packets,
including paired module/optimizer states from independent writers, module-only,
optimizer-only, and metadata-only messages with versions 0..3. All 20,439
possible ordered traces of lengths one through three were checked against
production `Learner.set_state` with controlled state holders. The minimal
patch yielded **zero violations of the specified causal admission and version
publication properties**. The old #67131 head failed two invariant classes;
the unchanged official Ray master commit `fbd093c` failed four invariant classes.
[Bounded proof experiment CI](https://github.com/lindicaphxag-tech/ray/actions/runs/39912446609).

This is bounded enumeration of sequential calls, not a proof over arbitrary
concurrency, network faults, failures during setters, or distributed ACKs.
Real Adam regressions and actual wheel-native TestLearner methods are separate
independent evidence. These tests do not establish an improvement in RL reward.

Both reviewed production and native-test files on
[the two-file minimal branch](https://github.com/lindicaphxag-tech/ray/compare/fix/rllib-version-unversioned-set-state-20261018...review/rllib-causal-paired-state-minimal-20261023)
have matching Git blob SHA values on the research test branch; no synthetic
proof scripts are part of the minimal patch.

Maintainer approval of equal-version authoritative restore semantics and
failure handling remains pending. #67131 also still has a failing Microcheck,
so no merge-ready or externally accepted claim follows from these CI results.
