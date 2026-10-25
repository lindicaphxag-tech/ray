# RLlib Learner snapshot admission is not exception-atomic

**Scope:** Fork-only fault injection against reviewed 3-component `Learner.set_state` patch, not official merged Ray, not a new optimizer, and not a claim of ACID transaction semantics. Independent external reviewer acceptance remains pending.

## A sharper failure than stale-version rejection

State is `S = (v, W, O, T)` for published weight sequence, installed RLModule, optimizer memory, and `should_module_be_updated` selector. The three-component candidate correctly ties all *successful-call* changes to one admission predicate, but on an **accepted v11 snapshot** its actual Python mutation order is:

1. `module.set_state(W11)`
2. `_set_optimizer_state(O11)`
3. `_weights_seq_no = 11`
4. `config.multi_agent(policies_to_train=T11)`
5. optional independent metrics state

Component setters can raise exceptions before or after mutating their respective state. There is no staging/rollback contract in this method.

- If the optimizer setter raises **after** writing, the receiver may be `(10, W11, O11, T10)`. The installed state is no longer described by version 10.
- If the selector setter raises **before** writing, the receiver may be `(11, W11, O11, T10)`. This is more serious: the receiver already advertises v11, and **replaying the same v11 packet** is rejected by the strict-newer admission check, so the selector remains T10.
- A version counter that is monotonic and a successful-call component admission predicate are therefore **not sufficient** to certify exception-atomic state publication.

### Verified six-stage actual-source experiment

[CI #39988732670](https://github.com/lindicaphxag-tech/ray/actions/runs/39988732670) executes the candidate's **actual `Learner.set_state` method AST**, not a handwritten reimplementation. Six injected before/after setter failures and a clean path confirm the two torn-state witnesses and same-version replay problem. A standalone fail-closed wrapper, separate from production, correctly refuses a next `train` or `set_state` after any observed setter failure. It **does not** repair the in-memory torn components and is not crash durable.

[Standalone executable](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_learner_commit_failure_fence.py)

### Native Ray extension

[Native actual-Learner experiment](https://github.com/lindicaphxag-tech/ray/blob/research/rllib-state-commit-failstop-20261024/ci/rllib_native_commit_failure_training.py) installs Ray 2.59.0 + CPU Torch, executes `Learner.set_state` from exact candidate source in a **real** built Learner, injects a selector setter failure after model/optimizer/version publication, and checks real tensor state and same-version replay. Its subsequent test of an actual CartPole training batch depends on the Ray source-tree offline `large.json` fixture not shipped in the PyPI wheel; the first native run established **changed model parameters (0.125), version publication to 2, and failed same-version recovery**, then stopped because the fixture was missing. The subsequent run copies that unchanged fixture into the native wheel before training. **Do not claim native training-step success until its exact CI completes.**

## Practical protocol design, not an implemented upstream remedy

A high-integrity alternative would separate `STAGING` and `COMMITTED` epochs and enforce:
- a received complete checkpoint has a declared component mask, writer/epoch identity, and explicit restore intent;
- a Learner that has begun a multi-component install must block training/snapshot publication until every required component completes;
- after any exception, mark the Learner **uncertain** and rebuild it from an authoritative complete state, rather than reusing the same version counter as proof of health;
- publish the new externally visible epoch and advertise it to EnvRunners only after all components and selector are confirmed installed;
- independently pin/attest rollout producer provenance and inference-cache epoch before accepting mixed-policy gradients.

Simply moving `_weights_seq_no = v` to the last line does **not** solve partial state mutation on a failing setter, and wrapping all setters in `try/except` without rollback or quarantine does not establish atomicity.

An opt-in fail-stop transaction boundary, or a true shadow-copy-and-swap strategy with explicit resource costs, requires maintainer API agreement and native multi-actor experiments. The current source-only fail-closed prototype protects operations that pass through that wrapper but does not intercept arbitrary other RLlib call sites.

## Scientific maturity and contribution hygiene

The underlying principles of two-phase state publication, fail-closed recovery, and snapshot consistency are established distributed-systems ideas. The new contribution candidate is a **specific independently reproduced RLlib failure in the presence of otherwise-correct causal-admission logic**, not a new general impossibility theorem or a measured gain in RL reward. A high-tier algorithm claim would additionally require cleanly specified writer identity, efficient cache/rule transfer, true recovery semantics, and independent end-to-end reward/throughput comparison with strong baselines.

**Upstream state:** issue [ray-project/ray#67292](https://github.com/ray-project/ray/issues/67292) open; [#67131](https://github.com/ray-project/ray/pull/67131) open; this is [fork Draft #11](https://github.com/lindicaphxag-tech/ray/pull/11), not an external Merge.


## Native gradient-step outcome (completed second run)

[Successful native Ray/PyTorch Actions #39989496133](https://github.com/lindicaphxag-tech/ray/actions/runs/39989496133) now includes the missing original Ray CartPole offline fixture. With the candidate's real `Learner.set_state` injected into an actually built `BaseTestingAlgorithmConfig().build_learner(env=CartPole-v1)`:

1. A deliberately modified RLModule state increased an actual floating model parameter by **0.125**, while the accepted incoming weight version advanced to **2**.
2. `config.multi_agent` raised a controlled error *before* installing the selector; the old selector remained active.
3. Replaying the identical version-2 complete checkpoint did **not** correct that selector because the new positive version was already installed.
4. The real Learner processed an actual CartPole batch of **200 rows** and updated the default policy's parameters by a measured max absolute change of **0.0010000020265579224**. The delivered selector intended to exclude that default policy. Returned result keys included `__all_modules__` and `default_policy`.

This is a deterministic causal **correctness** witness, not an assessment of aggregate episodic return or a generalization claim. The first native run separately confirmed steps 1–3 but could not load the CartPole dataset absent from the released wheel; the second run explicitly installed the unchanged source-tree fixture and successfully completed the training step. No production Ray code was changed by this experiment.

**Recommended next external discussion:** the correct response to an exception during multi-component state installation: fail-stop and reconstruct from an authoritative committed snapshot, or provide a true staged/swap checkpoint interface. Swapping version assignment order alone cannot prevent partial writes. The review candidate is not ready to claim exception-atomicity.
