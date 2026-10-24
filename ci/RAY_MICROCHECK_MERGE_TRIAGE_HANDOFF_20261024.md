# Ray upstream Microcheck — merge-unblocking triage handoff (2026-10-24)

> **Status:** Fork-only, documentation and external native validation. No upstream Buildkite job traceback is available from this integration, and no failure is classified as infrastructure without evidence. GitHub currently returns HTTP 403 for upstream comments/review requests via this integration. **Do not claim this handoff was posted to an official Ray PR.**

## Verified GitHub states (all Open, mergeable, no human approval)

| Official PR | Source head | Native independent evidence | Official Buildkite Microcheck |
|---|---|---|---|
| [Ray #67329](https://github.com/ray-project/ray/pull/67329) | `2a58ac235f7648e7776f17fdd192384fb976b68c` | [Actual PyPI Ray 2.59 native wheel + 6/6 full EnvRunner tests, with old-code deterministic RED](https://github.com/lindicaphxag-tech/ray/actions/runs/39982244111) | [#56979 — FAILURE](https://buildkite.com/ray-project/microcheck/builds/56979) |
| [Ray #67131](https://github.com/ray-project/ray/pull/67131) | `b845e62f77bc9fc6dc4e66f5093274e0c5933384` | [Native source and Learner validation #39955155935](https://github.com/lindicaphxag-tech/ray/actions/runs/39955155935), [#39955479007](https://github.com/lindicaphxag-tech/ray/actions/runs/39955479007) | [#56975 — FAILURE](https://buildkite.com/ray-project/microcheck/builds/56975) |
| [Ray #67103](https://github.com/ray-project/ray/pull/67103) | `7795d644673b4a4ed4d407fd9e7b0ca981e9606b` | [Source-exact callback/EnvRunner integration #39815603690](https://github.com/lindicaphxag-tech/ray/actions/runs/39815603690); official PR additionally describes native callback test correction | [#56950 — FAILURE](https://buildkite.com/ray-project/microcheck/builds/56950) |

The official Release and ReadTheDocs checks **pass** on all three exact heads. These positive checks do not waive Microcheck.

**Control, to avoid assuming a global outage:** unrelated [Ray #67328](https://github.com/ray-project/ray/pull/67328) is **Merged** with **Microcheck success**; [Ray #67331](https://github.com/ray-project/ray/pull/67331) is Open with **Microcheck success**. Thus it is unsafe to presume every Ray PR fails this CI merely because these three do. Conversely, differences in file paths and tests mean no causal conclusion follows from those other PRs.

## Narrowest-first maintainer ask

> I would appreciate help identifying the **first failed Buildkite Microcheck job name, its exact traceback, and whether it is related to the touched files** for #67329 (build 56979), #67131 (56975) and #67103 (56950). The integration can read the GitHub status contexts but does not expose Buildkite job-level logs. I do not want to churn code or request reruns without identifying an infra failure. In particular, #67329 is a narrowly scoped two-file EnvRunner version fix; the same exact production and test file ran under the real Ray 2.59.0 native extension/PyTorch wheel with all 6 scoped tests passing, while restoring just the old two-line condition made the regression test fail as expected ([receipt](https://github.com/lindicaphxag-tech/ray/actions/runs/39982244111)). If the Buildkite failure is from the PR, I can make the smallest targeted correction; if unrelated, please advise the appropriate rerun/CI ownership. Once checks are resolved, I would welcome an up-to-date human review of the exact head.

## Ordered local actions after actual traceback becomes visible

1. Determine whether the first failing job reproduces against the same base SHA and test command, or is deterministic against the patch only. Compare stack trace and touched files; **no blind rebase/squash/re-run**.
2. If it involves the fix, add one minimal native regression first, fail on old code, pass on amended branch, then update the exact official PR head **only after validating**.
3. If it is provably unrelated to changed code and an approved rerun procedure exists, request a fresh Microcheck from authorized CI maintainers.
4. After CI is green, request re-review on the latest exact head. Do not expand these PRs with the separate fork-only epoch-fencing / LearnerGroup research protocol.
5. Treat only GitHub `merged=true` in the external Ray repository as a Merge, never a fork Draft or independent CI status.

## Communication/permission obstacle

On 2026-10-24, direct upstream JAX reviewer-comment reply, upstream JAX issue comment and HF smolagents review request each produced `HTTP 403 Resource not accessible by integration` with the current GitHub connector. No successful official message from this handoff is claimed. A user-authenticated GitHub web session with permission to comment, or an authorized workflow/connector, is needed to deliver the request. No private Buildkite API tokens should be embedded in PR comments, files or CI workflows.
