"""Run actual RLlib Learner.set_state with genuine torch Adam state.

No Ray runtime is required: the *unchanged production method body* is compiled
from the checked-out source. Only Learner's unneeded runtime infrastructure
is replaced by a minimal adapter around a real torch.nn.Module and
torch.optim.Adam (including actual exp_avg/exp_avg_sq and step counters).

This is a deterministic transactional-state counterexample, not measured
reinforcement learning benchmark quality or a claim about intended semantics.
"""
from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def native_learner_set_state():
    path = Path(sys.argv[1]) if len(sys.argv)>1 and not sys.argv[1].startswith("--") else Path("python/ray/rllib/core/learner/learner.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    clazz = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Learner")
    method = copy.deepcopy(next(node for node in clazz.body if isinstance(node, ast.FunctionDef) and node.name == "set_state"))
    method.decorator_list = []
    method.returns = None
    method.args.args[1].annotation = None
    scope = {
        "COMPONENT_RL_MODULE": "module",
        "COMPONENT_OPTIMIZER": "optimizer",
        "COMPONENT_METRICS_LOGGER": "metrics",
        "WEIGHTS_SEQ_NO": "version",
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
    return scope["set_state"]


class ActualTorchModule:
    def __init__(self, parameter):
        self.parameter = parameter

    def set_state(self, state):
        with torch.no_grad():
            self.parameter.copy_(state["parameter"])


class LearnerAdapter:
    def __init__(self, weights, opt_state, version):
        self.parameter = torch.nn.Parameter(weights.clone())
        self.module = ActualTorchModule(self.parameter)
        self.optimizer = torch.optim.Adam([self.parameter], lr=0.03)
        self.optimizer.load_state_dict(copy.deepcopy(opt_state))
        self._weights_seq_no = version
        self.config = SimpleNamespace(multi_agent=lambda **kwargs: None)
        self.metrics = SimpleNamespace(set_state=lambda *_: None)

    def _check_is_built(self):
        pass

    def _set_optimizer_state(self, state):
        self.optimizer.load_state_dict(copy.deepcopy(state))

    def snapshot(self):
        return {
            "version": self._weights_seq_no,
            "module": {"parameter": self.parameter.detach().clone()},
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def training_step(self, target):
        self.optimizer.zero_grad()
        loss = ((self.parameter - torch.as_tensor(target, dtype=torch.float64)) ** 2).mean()
        loss.backward()
        self.optimizer.step()


torch.manual_seed(24681)
baseline = "--baseline" in sys.argv
native_method = native_learner_set_state()
trainable = torch.nn.Parameter(torch.tensor([0.5, -1.25], dtype=torch.float64))
original_optimizer = torch.optim.Adam([trainable], lr=0.03)

for step in range(9):
    original_optimizer.zero_grad()
    target = torch.tensor([1.3 + step * .11, -.3 + step * .06], dtype=torch.float64)
    ((trainable - target) ** 2).mean().backward()
    original_optimizer.step()

checkpoint_v9 = {
    "version": 9,
    "module": {"parameter": trainable.detach().clone()},
    "optimizer": copy.deepcopy(original_optimizer.state_dict()),
}
original_optimizer.zero_grad()
((trainable - torch.tensor([2.3, .4], dtype=torch.float64)) ** 2).mean().backward()
original_optimizer.step()
checkpoint_v10 = {
    "version": 10,
    "module": {"parameter": trainable.detach().clone()},
    "optimizer": copy.deepcopy(original_optimizer.state_dict()),
}
assert checkpoint_v9["optimizer"]["state"][0]["step"].item() == 9
assert checkpoint_v10["optimizer"]["state"][0]["step"].item() == 10

correct = LearnerAdapter(
    checkpoint_v10["module"]["parameter"], checkpoint_v10["optimizer"], 10
)
stale_replay = LearnerAdapter(
    checkpoint_v10["module"]["parameter"], checkpoint_v10["optimizer"], 10
)
native_method(stale_replay, checkpoint_v9)

assert stale_replay._weights_seq_no == 10
assert torch.equal(stale_replay.parameter, correct.parameter)
assert correct.optimizer.state_dict()["state"][0]["step"].item() == 10
if baseline:
    assert stale_replay.optimizer.state_dict()["state"][0]["step"].item() == 9
    print("EXPECTED BASELINE: real Adam step=9 loaded while module and version remain v10")
else:
    assert stale_replay.optimizer.state_dict()["state"][0]["step"].item() == 10
    print("PASS PATCH: reject stale paired optimizer, preserving Adam step v10")

correct.training_step([2.6, .7])
stale_replay.training_step([2.6, .7])
weight_error = torch.linalg.vector_norm(correct.parameter - stale_replay.parameter).item()
if baseline:
    assert weight_error > 1e-5, weight_error
    print(f"EXPECTED BASELINE: next Adam step diverges from v10 by L2={weight_error:.12f}")
else:
    assert weight_error < 1e-12, weight_error
    print(f"PASS PATCH: identical next Adam update to consistent v10 (L2={weight_error:.12f})")

# Distinct, legitimate API: an optimizer-only state update must be regarded
# separately; a proposed paired-restore policy must not silently change it.
only_optimizer = LearnerAdapter(
    checkpoint_v10["module"]["parameter"], checkpoint_v10["optimizer"], 10
)
native_method(only_optimizer, {
    "version": 9, "optimizer": copy.deepcopy(checkpoint_v9["optimizer"])
})
assert only_optimizer.optimizer.state_dict()["state"][0]["step"].item() == 9
assert only_optimizer._weights_seq_no == 10
print("PASS: optimizer-only API's existing behavior is independently identified")

# Fresh paired snapshot is accepted and restores both optimizer and weights.
fresh = LearnerAdapter(
    checkpoint_v9["module"]["parameter"], checkpoint_v9["optimizer"], 9
)
native_method(fresh, checkpoint_v10)
assert fresh._weights_seq_no == 10
assert torch.equal(fresh.parameter, checkpoint_v10["module"]["parameter"])
assert fresh.optimizer.state_dict()["state"][0]["step"].item() == 10
print("PASS: newer paired snapshot restores both live weight and Adam state")

# Control: unversioned weight mutation maintains existing PR #67131 semantics.
unversioned = LearnerAdapter(
    checkpoint_v10["module"]["parameter"], checkpoint_v10["optimizer"], 10
)
native_method(unversioned, {
    "module": {"parameter": checkpoint_v9["module"]["parameter"]},
})
assert unversioned._weights_seq_no == 11
print("PASS: unversioned module-only mutation remains visible as v11")
