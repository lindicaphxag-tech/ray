# Noninvasive checkpoint-continuation falsification gate

The existing DDP gradient/Adam audit on Fork PR #11 was **red**, with an observed next-step mismatch despite equivalent checkpoint model and optimizer buffers and gradient hashes. Later instrumentation around `torch.optim.Adam.step` itself yielded rank-dependent disagreement between original optimizer and cloned one, so **the hypothesis that Ray is defective was not established**. This protocol avoids modifying optimizer or gradient methods entirely.

## Native protocol
- Installed `ray[rllib]==2.59.0` + CPU PyTorch, built native `BaseTestingAlgorithmConfig().build_learner`.
- Same CartPole offline batch, deterministic PyTorch/NumPy/Python RNG resets, four prespecified seeds: 7, 19, 31, 43.
- One native warm-up update to populate real optimizer moments; deepcopy the full `Learner.get_state` checkpoint.
- Construct a second fresh native Learner, apply that exact checkpoint with `set_state`.
- Compare checkpoint's numeric RLModule state, numeric optimizer buffers, AND attached live Adam moment buffers **before training**.
- Execute one normal uninstrumented native `Learner.update` on each instance, with same batch/RNG; compare model state, optimizer state, live Adam moments.
- Output each seed's discrepancies unconditionally; do not assert that the desired equality must hold. The result is allowed to falsify the hypothesis.

No actual Ray Actor kill, DDP interaction, optimizer monkeypatch, or algorithmic remedy is claimed in this new minimal experiment. This test is a *negative control* to distinguish intrinsic checkpoint continuation discrepancies from the earlier invasive DDP debugging harness. If it passes, reintroduce DDP and process death one factor at a time before alleging a product bug. If it fails, isolate a minimal PyTorch optimizer state or Ray Learner serialization cause and benchmark a verified fix.

The experiment modifies only the research branch; official upstream #67131 / #67329 remain untouched.
