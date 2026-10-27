# RLlib optimizer ownership: make the Bazel test actually execute (2026-10-27)

**Fork-only corrective evidence**, not a submitted or merged upstream PR. Based directly on [Ray #67351](https://github.com/ray-project/ray/pull/67351) at `bffd56ffda7f5e1f1211168de7ac42d2ecf9d850`.

## Finding — a Bazel target can pass without running its test

The early Cursor review comment said the new optimizer regression was missing registration in `BUILD.bazel`. That portion is already **fixed** in the official PR: `python/ray/rllib/BUILD.bazel` contains a `py_test` target named `test_torch_optimizer_restore_ownership` referencing `core/learner/tests/test_torch_optimizer_restore_ownership.py`.

The remaining problem is that the official test module has only a top-level pytest test function, but **no `if __name__ == "__main__": pytest.main(...)` entrypoint**. The nearby registered `core/learner/torch/tests/test_torch_learner_optimizer_restore.py` uses the entrypoint with `sys.exit(pytest.main(["-v", __file__]))`. Ray's `BUILD.bazel` imports `py_test` from `@rules_python//python:defs.bzl`: the Bazel launcher executes the target Python file, not pytest collection, unless the file explicitly dispatches pytest. The absent entrypoint can therefore produce false-green Bazel results after merely importing and defining the test.

**This is a separate *test execution* defect**, not the original optimizer storage-alias root cause; correctness evidence for the production method remains recorded in [prior real native CPU A/B](https://github.com/lindicaphxag-tech/ray/actions/runs/40021945290).

## Minimal staged fix

1. [One 108-byte appended entrypoint](../python/ray/rllib/core/learner/tests/test_torch_optimizer_restore_ownership.py) copies the style of Ray's existing optimizer test and propagates pytest's exit status:
   ```python
   if __name__ == "__main__":
       import sys
       import pytest
       sys.exit(pytest.main(["-v", __file__]))
   ```
2. [Pure stdlib Bazel execution audit](../ci/check_optimizer_ownership_bazel_entry.py) confirms the BUILD target, test function, pytest dispatch and exit propagation; removing the entrypoint serves as the expected RED case.
3. [Native Ray 2.59 stock-red/source-exact-patched-green workflow](../.github/workflows/rllib_optimizer_bazel_entry_native_20261027.yml) directly invokes the copied test file as Bazel would. It requires the unpatched wheel to exit **1 for the known checkpoint alias assertion**, and the patched same test to exit **0**. The source-exact method loader is reused from an earlier successful fork workflow and overlays only the changed method to avoid unrelated API mismatches with published Ray.

**Verification boundary:** current official-vs-fork file reads confirm original missing entrypoint and new fixed entrypoint, while current BUILD target registration is confirmed. The new GitHub Actions workflow has been pushed but **no new run is yet visible**. Therefore the workflow is not described as passed; static proof of direct entrypoint is not a substitute for the native Ray test passing with latest code.

## Why not edit the upstream PR immediately?

Official Ray #67351 still has **three author commits failing DCO sign-off**. The additional test-entrypoint change is staged on this **separate research branch** so it does not create yet another unsigned official commit. DCO sign-off is an author attestation and must be performed by the real contributor, including these existing commits, before the combined fix is an eligible merge candidate. After a valid author sign-off, bring in this entrypoint as a separately signed commit and rerun the native check. The Buildkite Microcheck failure remains independently unresolved without its accessible detailed logs.

**Attribution:** authored on fork branch `research/rllib-optimizer-bazel-test-entry-20261027`; no adoption or upstream approval claimed.
