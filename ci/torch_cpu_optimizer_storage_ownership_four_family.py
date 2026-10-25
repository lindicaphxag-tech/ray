"""Reproducible CPU optimizer-checkpoint storage ownership counterfactual.

Research-only; not a Ray production patch or proof of DDP continuation parity.
Requires numpy + torch. Uses WRITABLE NumPy checkpoint arrays to avoid
undefined behavior from writing through torch.from_numpy(readonly_array).
The read-only Ray object-store case may be more dangerous, not less.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import torch


def check_family(name: str) -> dict:
    options = {
        "Adam": {},
        "AdamW": {},
        "RMSprop": {"momentum": 0.9},
        "SGD": {"momentum": 0.9},
    }[name]
    optimizer_type = getattr(torch.optim, name)
    base = torch.nn.Parameter(torch.tensor([0.7, -1.2], dtype=torch.float32))
    first = optimizer_type([base], lr=0.01, **options)
    base.grad = torch.tensor([0.2, -0.1])
    first.step()
    snapshot = first.state_dict()
    frozen_arrays = {
        index: {
            key: value.detach().clone().numpy().copy()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
            for key, value in buffers.items()
        }
        for index, buffers in snapshot["state"].items()
    }

    arms = []
    for make_private in (False, True):
        source = copy.deepcopy(frozen_arrays)
        parameter = torch.nn.Parameter(base.detach().clone())
        optimizer = optimizer_type([parameter], lr=0.01, **options)
        converted = {
            index: {
                key: (
                    (torch.from_numpy(value).clone() if make_private
                     else torch.from_numpy(value))
                    if isinstance(value, np.ndarray)
                    else copy.deepcopy(value)
                )
                for key, value in buffers.items()
            }
            for index, buffers in source.items()
        }
        optimizer.load_state_dict({**snapshot, "state": converted})

        # Inspect the *loaded optimizer*, not only the conversion input.
        aliases = []
        for index, buffers in optimizer.state_dict()["state"].items():
            for key, value in buffers.items():
                original = source[index].get(key)
                if isinstance(value, torch.Tensor) and isinstance(original, np.ndarray):
                    if value.data_ptr() == original.__array_interface__["data"][0]:
                        aliases.append(key)

        prior = copy.deepcopy(source)
        parameter.grad = torch.tensor([0.13, -0.22])
        optimizer.step()
        modified = [
            key for index, buffers in source.items()
            for key, value in buffers.items()
            if isinstance(value, np.ndarray)
            and not np.array_equal(value, prior[index][key])
        ]
        arms.append((aliases, modified, parameter.detach().clone()))

    unowned, owned = arms
    maximum_weight_error = float(
        (unowned[2] - owned[2]).abs().max().item()
    )
    assert unowned[0], f"{name}: expected at least one zero-copy storage alias"
    assert unowned[1], f"{name}: expected a source checkpoint mutation"
    assert not owned[0], f"{name}: copied restore still aliases source"
    assert not owned[1], f"{name}: copied restore mutated source"
    assert maximum_weight_error == 0.0, (
        f"{name}: ownership copy changed next-step numerical result"
    )
    return {
        "optimizer": name,
        "unowned_alias_keys": sorted(unowned[0]),
        "unowned_mutated_checkpoint_keys": sorted(unowned[1]),
        "owned_alias_keys": owned[0],
        "owned_mutated_checkpoint_keys": owned[1],
        "one_step_model_max_abs_difference": maximum_weight_error,
    }


if __name__ == "__main__":
    print("VERSIONS", json.dumps({
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
    }))
    for family in ("Adam", "AdamW", "RMSprop", "SGD"):
        print("CPU_OWNERSHIP_ABLATION", json.dumps(check_family(family), sort_keys=True))
    print("FOUR_OPTIMIZER_OWNERSHIP_CONTROLS_PASSED")
