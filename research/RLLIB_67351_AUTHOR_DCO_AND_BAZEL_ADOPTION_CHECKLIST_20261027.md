# Ray #67351 author-only DCO repair + tested Bazel entrypoint adoption

Upstream PR: https://github.com/ray-project/ray/pull/67351
Official fork head as verified: bffd56ffda7f5e1f1211168de7ac42d2ecf9d850
Research-only target branch: research/rllib-optimizer-bazel-test-entry-20261027

## Verify the existing hard block

The official PR has exactly three user-authored commits. Each has `Signed-off-by` absent:
- be7729f (production CPU optimizer-owned copy)
- 9121dd5 (native optimizer alias regression)
- bffd56f (BUILD.bazel registration/test formatting)

Three missing signed trailers, independent of Buildkite microcheck failure.

## Commands for the actual human contributor (not to be executed by an agent)

These commands **make a legal Developer Certificate of Origin attestation**. Only run them after personally reviewing every changed line and confirming the rights to submit the code. Verify the Git author name/email are correctly configured first. They **rewrite the last three commit hashes**, which also requires safe force-push after inspecting divergence; do not run on a shared or unclean branch.

```bash
git clone https://github.com/lindicaphxag-tech/ray.git
cd ray
git fetch origin
git checkout fix/rllib-optimizer-actor-owned-buffers-20261024
git status --short
git log -3 --format='%h %an <%ae>%n%s%n%b'
# Stop if git status is not clean, author identity is wrong, or head differs.
git rev-parse HEAD  # expected bffd56ffda7f5e1f1211168de7ac42d2ecf9d850
git config user.name
git config user.email
# Contributor-only, after they accept the DCO:
git rebase --exec 'git commit --amend --no-edit --signoff' HEAD~3
git log -3 --format='%h %s%n%b'
```

If the DCO is correct and history is safe to publish, bring in the already prepared missing test runner (single 108-byte file change) **as a separate signed commit**:

```bash
git cherry-pick --no-commit 9d5fcda305fd2893bd21e8d69b551d6bee4b64ef
git diff --cached --check
git diff --cached
git commit -s -m 'test(rllib): execute optimizer ownership pytest in Bazel'
# Run the focused native tests; the scripts on the research branch can assist.
git push --force-with-lease origin HEAD:fix/rllib-optimizer-actor-owned-buffers-20261024
```

Use an explicit fetch of the research branch before cherry-pick if not yet fetched: `git fetch origin research/rllib-optimizer-bazel-test-entry-20261027`.

## What remains after signing

- Confirm updated official PR reports a green DCO check; GitHub status may take time to update.
- Verify that the direct file script fails on stock Ray and passes with the production `_set_optimizer_state` patch (not merely `python -m pytest`).
- Inspect **actual** Buildkite microcheck failure, identify affected targets, and ask a Ray maintainer for `go` full-suite review only after the targeted tests pass.
- Human author must confirm understanding and local test execution per Ray `AGENTS.md` AI-assisted PR contribution policy; do not interpret this plan as author consent.

## Audit boundary

The assistant has NOT signed, rewritten, or force-pushed author commits and has NOT made any official PR update. Fork preparation and local PyTorch / structural tests cannot replace DCO and native CI.