"""Independent CPU PyTorch cross-optimizer checkpoint storage witness.

Reproduces actual Optimizer.load_state_dict alias/contamination for Adam,
AdamW and momentum-SGD (float32/float64) when two receivers share a single
NumPy-backed sender checkpoint. No Ray actor or installed Ray is involved.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import torch


CASES = {
    "Adam": lambda p: torch.optim.Adam([p], lr=0.01),
    "AdamW": lambda p: torch.optim.AdamW([p], lr=0.01),
    "SGD_momentum": lambda p: torch.optim.SGD([p], lr=0.01, momentum=0.9),
}


def new_optimizer(kind, base):
    parameter = torch.nn.Parameter(base.clone())
    return parameter, CASES[kind](parameter)


def step(parameter, optimizer, grad):
    parameter.grad = grad.clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def checkpoint_numpy(optimizer):
    state = copy.deepcopy(optimizer.state_dict())
    for slots in state["state"].values():
        for name, tensor in slots.items():
            if torch.is_tensor(tensor):
                slots[name] = tensor.detach().cpu().numpy().copy()
    return state


def restore(kind, base, state, own):
    parameter, optimizer = new_optimizer(kind, base)
    raw = {"param_groups": copy.deepcopy(state["param_groups"]), "state": {}}
    for index, slots in state["state"].items():
        raw["state"][index] = {}
        for name, value in slots.items():
            raw["state"][index][name] = (
                torch.from_numpy(value).clone() if own else torch.from_numpy(value)
            ) if isinstance(value, np.ndarray) else copy.deepcopy(value)
    optimizer.load_state_dict(raw)
    return parameter, optimizer


def trial(kind, dtype):
    base = torch.tensor([0.5, -1.2, 1.7], dtype=dtype)
    grad0 = torch.tensor([0.2, -0.3, 0.4], dtype=dtype)
    grad1 = torch.tensor([-0.8, 0.6, -0.2], dtype=dtype)
    warm_parameter, warm_optimizer = new_optimizer(kind, base)
    step(warm_parameter, warm_optimizer, grad0)
    weights = warm_parameter.detach().clone()
    torch_state = copy.deepcopy(warm_optimizer.state_dict())
    sender = checkpoint_numpy(warm_optimizer)
    buffers = sender["state"][0]
    originals = {
        name: value.copy() for name, value in buffers.items()
        if isinstance(value, np.ndarray)
    }

    reference_parameter, reference_optimizer = new_optimizer(kind, weights)
    reference_optimizer.load_state_dict(copy.deepcopy(torch_state))
    step(reference_parameter, reference_optimizer, grad1)

    bad_p1, bad_o1 = restore(kind, weights, sender, own=False)
    bad_p2, bad_o2 = restore(kind, weights, sender, own=False)
    aliased = [
        name for name in originals
        if bad_o1.state[bad_p1][name].data_ptr() == buffers[name].ctypes.data
    ]
    step(bad_p1, bad_o1, grad1)
    sender_changed = [
        name for name, prior in originals.items()
        if not np.array_equal(prior, buffers[name])
    ]
    step(bad_p2, bad_o2, grad1)
    bad_error = float(
        torch.max(torch.abs(bad_p2.detach() - reference_parameter.detach()))
    )

    for name, prior in originals.items():
        buffers[name][...] = prior
    good_p1, good_o1 = restore(kind, weights, sender, own=True)
    good_p2, good_o2 = restore(kind, weights, sender, own=True)
    step(good_p1, good_o1, grad1)
    step(good_p2, good_o2, grad1)
    error1 = float(
        torch.max(torch.abs(good_p1.detach() - reference_parameter.detach()))
    )
    error2 = float(
        torch.max(torch.abs(good_p2.detach() - reference_parameter.detach()))
    )
    state_errors = {
        name: float(torch.max(torch.abs(
            good_o2.state[good_p2][name]
            - reference_optimizer.state[reference_parameter][name]
        ))) for name in originals
    }
    sender_unchanged = all(
        np.array_equal(buffers[name], prior)
        for name, prior in originals.items()
    )
    assert (
        sender_unchanged and error1 == 0 and error2 == 0
        and all(v == 0 for v in state_errors.values())
    ), (kind, dtype, error1, error2, state_errors)
    return {
        "optimizer": kind,
        "dtype": str(dtype),
        "aliased": aliased,
        "sender_changed": sender_changed,
        "bad_next_step_max_error": bad_error,
        "corrected_next_step_max_error": error2,
        "corrected_state_max_error": max(state_errors.values()),
        "source_unchanged": sender_unchanged,
    }


def main():
    results = []
    for kind in CASES:
        for dtype in (torch.float32, torch.float64):
            record = trial(kind, dtype)
            results.append(record)
            print(json.dumps(record), flush=True)
    assert all(row["aliased"] and row["sender_changed"] for row in results)
    assert all(row["bad_next_step_max_error"] > 0 for row in results)
    print("PASS all six cross-optimizer CPU ownership checks")
    print("Versions:", torch.__version__, np.__version__)
    return results


if __name__ == "__main__":
    main()
