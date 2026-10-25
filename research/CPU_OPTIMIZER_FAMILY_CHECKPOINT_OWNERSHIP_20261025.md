# CPU optimizer checkpoint ownership: cross-family causal check (2026-10-25)

**Scope:** standalone PyTorch CPU evidence; not a Ray integrated CI result, not a checkpoint replay theorem, and not a new external merge. Reproduction: [four-family executable](../ci/torch_cpu_optimizer_storage_ownership_four_family.py). An optional [Actions workflow](../.github/workflows/torch_cpu_optimizer_ownership_four_family.yml) is configured, but its run result has not yet been confirmed.

## Concrete counterfactual

For a writable NumPy checkpoint buffer `C`, if optimizer restore passes `torch.from_numpy(C)` without a copy, the restored optimizer may hold a tensor `T` whose storage equals `C`. In-place optimizer step can then mutate both `T` and the *source checkpoint* buffer, violating source snapshot immutability. On CPU `T_private = torch.from_numpy(C).clone()`, the optimizer reads identical numeric state but owns independent storage.

This is an **ownership invariant**, not a proof that CPU zero-copy necessarily changes next-step weights in a particular run. The stronger architectural implication is that shared-read-only checkpoint transfer requires ownership isolation at the consuming Learner boundary.

## Independently executed local witness

Environment: torch **2.10.0+cpu**, NumPy **2.3.5**, one CPU trainable tensor per optimizer; run one warm step to populate buffers, restore two independent optimizer instances from identical NumPy state arrays, then apply the same following gradient. Source arrays were deliberately **writable** so the witness avoids undefined behavior of PyTorch updates on read-only `torch.from_numpy` buffers.

| Optimizer | Unowned aliased state keys | Checkpoint keys mutated by next step | Owned-copy aliases / source mutations | Next-step parameter max difference |
|---|---|---|---|---:|
| Adam | step, exp_avg, exp_avg_sq | same three | none / none | 0 |
| AdamW | step, exp_avg, exp_avg_sq | same three | none / none | 0 |
| RMSprop (momentum 0.9) | step, square_avg, momentum_buffer | same three | none / none | 0 |
| SGD (momentum 0.9) | momentum_buffer | same | none / none | 0 |

The entire table was observed from actual PyTorch `Optimizer.load_state_dict` and `step` calls, not inferred from the presence of `torch.from_numpy` alone. **Local proof-of-concept does not validate all Ray optimizers, devices, versions or distributed checkpoint pathways.**

## Direct link to reviewable upstream contribution

Official bounded fix: [ray-project/ray#67351](https://github.com/ray-project/ray/pull/67351) copies converted **CPU per-parameter optimizer buffers** before loading state and adds one native Ray regression / Bazel target. Its independent [native wheel red/green](https://github.com/lindicaphxag-tech/ray/actions/runs/40021945290) identified a real stock Adam buffer alias and passed after the targeted source patch.

In that red/green job, both the stock and patched two-Learner trajectories happened to match the reference next step; this witness therefore supports storage isolation, **not** a universally proven improvement in continuation error. Another independent run exhibited a stock-vs-copy trajectory difference, so replicate and control the runtime before attributing causality beyond the buffer mutation itself.

**Review strategy:** Do not add four optimizer families to the official Ray PR just to increase test count. The single real native Adam regression is a stronger integration lock for the exact production patch. Use this multi-family witness to justify *why* all restored mutable CPU optimizer buffers require private ownership, and keep the production change minimal until the failing Buildkite Microcheck traceback is available.
