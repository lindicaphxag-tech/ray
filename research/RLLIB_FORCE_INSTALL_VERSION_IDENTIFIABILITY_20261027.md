# RLlib monotonic weight sequence ≠ certified policy provenance

**2026-10-27 | fork-only mechanism audit, not a submitted upstream PR.**

Related official [Ray #67329](https://github.com/ray-project/ray/pull/67329) fixes a specific **positive-version rollback** in `MultiAgentEnvRunner.set_state`. The same small repair is staged for the `SingleAgentEnvRunner` on this fork branch. Do not generalize the accepted-number invariant into a claim that the scalar version alone identifies the resident weights after *all* state updates.

## Two observationally indistinguishable histories

Start with `(_weights_seq_no=8, resident_weights=W8)`.

1. **History A:** no later update. Public `WEIGHTS_SEQ_NO=8`, resident weights W8.
2. **History B:** `set_state({COMPONENT_RL_MODULE: W_other})` without `WEIGHTS_SEQ_NO`. The existing documented compatibility path force-installs W_other, but does **not** advance the public positive sequence. Public `WEIGHTS_SEQ_NO=8`, resident weights W_other.

These histories present the **same scalar public version** while their policy parameters may be different. Therefore no deterministic consumer that observes *only* `WEIGHTS_SEQ_NO` can infer a unique resident parameter snapshot across an unversioned force-install. This follows from two possible transitions mapping to the same observation, not from measurements of RL training performance.

The effect is present in the **old and fixed** `SingleAgentEnvRunner` and `MultiAgentEnvRunner` because the force-install rule is intentionally preserved. It does not invalidate the *narrow* fix: an **explicit positive stale** v7 update must not regress an existing positive v8 sequence.

## Executable witness and boundaries

[Cross-runner source-extracted red/green permutation checker](../ci/rllib_dual_env_runner_version_permutation_contract.py) now:
- Ensures each of 80 randomized, fixed-seed 24-step sequences explicitly covers old, equal, newer and unversioned/zero forced update cases (not just by chance);
- Checks 82 sequences / 1,935 transitions per runner, for 3,870 source-equivalent state transitions in total;
- Confirms a historical v7 rollback as RED and the guarded monotonic version assignment as GREEN;
- Shows an **EXPECTED LIMITATION** for both runner types: two different forced module payloads install successfully while `WEIGHTS_SEQ_NO` remains 8.

An independent local smoke test used faithful copied `set_state` method bodies for both runners and obtained the expected RED/GREEN and limitation results. This is **not** a certified hash-identical native Ray checkout or a passed native GitHub Actions run. The checked-in script reads the actual methods from the branch when executed there; no Ray wheel is required for that AST-level audit.

## Why this matters for cross-framework state lineage

Ray's single positive version, TRL's AsyncGRPO scheduling-time version ([issue #7861](https://github.com/huggingface/trl/issues/7861)), and token prefix-KV epoch labels address different events: accepted parameter state, requested generation state and cache-producing state. A single label is insufficient when state changes can occur without that label being updated. The proper scope for a future **provenance protocol** would define an explicit `known(epoch)` versus `unknown/forced` state, and certify the relevant request/serving/weights association before admitting asynchronous groups. That is a **research option**, not a new upstream API proposal or accepted implementation.

## Merge recommendation

Finish the narrow DCO-green Ray #67329 without expanding semantics; keep the single-agent 2-line analogue staged until native CI/maintainer agreement. Do not change forced-install semantics in that PR. Explicitly document that `WEIGHTS_SEQ_NO` is a monotonic **positive-version acceptance label under a restricted transition contract**, not an unconditional cryptographic or causal certificate of parameters.
