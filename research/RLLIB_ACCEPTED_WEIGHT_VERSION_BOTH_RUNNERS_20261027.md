# Ray RLlib: cross-runner accepted-weight version monotonicity (2026-10-27)

**Status:** fork-only follow-up. Not merged upstream, no new official PR created. Based on the exact head [Ray #67329 at `2a58ac2`](https://github.com/ray-project/ray/pull/67329/commits/2a58ac235f7648e7776f17fdd192384fb976b68c); upstream PR #67329 remains unchanged and DCO-green.

## Shared invariant

Whenever a positive-version RLModule payload is rejected because its version is not newer, the publicly advertised `WEIGHTS_SEQ_NO` must not regress. Both the already-patched `MultiAgentEnvRunner` (#67329) and the independently observed `SingleAgentEnvRunner` execute near-identical state transitions.

**Official latest master counterexample (SingleAgentEnvRunner):**

- Resident weights and sequence both at version 8.
- Receive stale version 7, reject `module.set_state`, but old code blindly assigns `_weights_seq_no = 7`.
- Receiving equal-to-resident v8 then spuriously reapplies weights because the runner now reports v7.

The single-agent path has an additional `DEFAULT_MODULE_ID` dictionary unwrapping contract absent from the multi-agent path. The new native-code test asserts this also stays unchanged. A missing/zero version continues to force-apply without advancing the globally published sequence.

## Patch

1. [SingleAgentEnvRunner production implementation](../python/ray/rllib/env/single_agent_env_runner.py): only update `_weights_seq_no` if the incoming positive version is greater than the currently published one.
2. [Test in the existing single-agent suite](../python/ray/rllib/env/tests/test_single_agent_env_runner.py): four deterministic cases (stale v7, equal v8, genuinely newer v9, unversioned forced update), using the real method and `DEFAULT_MODULE_ID` mapping.
3. [Source-extracted CPU red/green verifier](../ci/rllib_single_agent_weight_version_contract.py): extracts and compiles the exact patched `set_state` AST from the checked-out source, then reverts just the monotonic version condition for a matched negative control. **No Ray wheels, actor startup or GPU required.**

## Evidence / limitations

A source-equivalent local prototype of the four-case harness returned:

```
EXPECTED RED: old source gate fails when a stale v7 update makes equal v8 appear new
GREEN: patched source gate passes stale/equal/new/unversioned four-case contract
```

The exact AST extraction script was independently smoke-tested locally using a fixture that reproduces the production method body; it is published, **but no exact-checkout Actions run or full native RLlib test pass is claimed here**. Native Ray wheel regression, bazel test registration and integration CI are still needed for adoption.

This is not a new numerical algorithm or proof of distributed linearizability; it is a consistent state-provenance repair across both supported EnvRunner implementations. It does not assert that real actors reordered messages in deployed training.

## Merge hygiene

[Ray #67329](https://github.com/ray-project/ray/pull/67329) already has two correctly signed-off author commits and a source-level plus native-wheel regression for the multi-agent case. **Do not silently add unsigned commits to that official PR**, because Ray's Developer Certificate of Origin requires the real contributor's sign-off on every commit. Maintainer review should determine whether to expand #67329 to both runner types or create a focused additional PR after native confirmation; do not inflate the official PR's status before that decision.

[Ray #67351](https://github.com/ray-project/ray/pull/67351) has a *separate* CPU optimizer-state aliasing invariant and three previously unsigned commits; the DCO problem there must be fixed by the actual author. Neither this fork branch nor its source-level tests remove #67351's DCO/Buildkite blockers.
