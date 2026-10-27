"""CPU-only Adam checkpoint sharing red/green, without Ray installation.

Pure PyTorch mechanism for Ray #67351; not a native RLlib integration test.
Run: python ci/rllib_numpy_adam_two_receiver_storage_witness.py
"""
import copy

import numpy as np
import torch


def make_optimizer(weights):
    parameter = torch.nn.Parameter(weights.detach().clone())
    return parameter, torch.optim.Adam([parameter], lr=0.01)


def take_step(parameter, optimizer, grad):
    parameter.grad = grad.clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def to_numpy_state(checkpoint):
    state = copy.deepcopy(checkpoint)
    for slots in state["state"].values():
        for name, value in tuple(slots.items()):
            if torch.is_tensor(value):
                slots[name] = value.detach().cpu().numpy().copy()
    return state


def restore_receiver(weights, checkpoint, *, own):
    parameter, optimizer = make_optimizer(weights)
    loaded = {
        "param_groups": copy.deepcopy(checkpoint["param_groups"]),
        "state": {
            idx: {
                key: (torch.from_numpy(v).clone() if own
                      else torch.from_numpy(v))
                if isinstance(v, np.ndarray) else copy.deepcopy(v)
                for key, v in slots.items()
            }
            for idx, slots in checkpoint["state"].items()
        },
    }
    optimizer.load_state_dict(loaded)
    return parameter, optimizer


def main():
    init = torch.tensor([0.5, -1.0, 1.5], dtype=torch.float32)
    grad0 = torch.tensor([0.2, -0.6, 0.4])
    grad1 = torch.tensor([-0.5, 0.3, -0.2])
    warm_p, warm_opt = make_optimizer(init)
    take_step(warm_p, warm_opt, grad0)
    weights = warm_p.detach().clone()
    warm_state = copy.deepcopy(warm_opt.state_dict())
    sender = to_numpy_state(warm_state)
    buffers = {
        name: value for name, value in sender["state"][0].items()
        if isinstance(value, np.ndarray)
    }
    assert {"step", "exp_avg", "exp_avg_sq"}.issubset(buffers)
    original = {name: arr.copy() for name, arr in buffers.items()}

    reference_p, reference_o = make_optimizer(weights)
    reference_o.load_state_dict(copy.deepcopy(warm_state))
    take_step(reference_p, reference_o, grad1)

    # Old path: two independently created Adam optimizers share one sender.
    bad_p1, bad_o1 = restore_receiver(weights, sender, own=False)
    bad_p2, bad_o2 = restore_receiver(weights, sender, own=False)
    assert (bad_o1.state[bad_p1]["exp_avg"].data_ptr()
            == bad_o2.state[bad_p2]["exp_avg"].data_ptr()
            == buffers["exp_avg"].ctypes.data)
    take_step(bad_p1, bad_o1, grad1)
    changed = [
        name for name, arr in buffers.items()
        if not np.array_equal(arr, original[name])
    ]
    assert set(changed) == {"step", "exp_avg", "exp_avg_sq"}, changed
    take_step(bad_p2, bad_o2, grad1)
    param_error = float(
        (bad_p2.detach() - reference_p.detach()).abs().max()
    )
    moment_error = float(
        (bad_o2.state[bad_p2]["exp_avg"]
         - reference_o.state[reference_p]["exp_avg"]).abs().max()
    )
    assert param_error > 0 and moment_error > 0
    print("EXPECTED RED: mutates", changed,
          "parameter_error", param_error, "moment_error", moment_error)

    # Copy all three state tensors, including Adam's scalar step counter.
    for name, arr in buffers.items():
        arr[...] = original[name]
    good_p1, good_o1 = restore_receiver(weights, sender, own=True)
    good_p2, good_o2 = restore_receiver(weights, sender, own=True)
    for name in ("step", "exp_avg", "exp_avg_sq"):
        first = good_o1.state[good_p1][name]
        second = good_o2.state[good_p2][name]
        assert first.data_ptr() != second.data_ptr()
        assert first.data_ptr() != buffers[name].ctypes.data
        assert second.data_ptr() != buffers[name].ctypes.data
    take_step(good_p1, good_o1, grad1)
    take_step(good_p2, good_o2, grad1)
    for p, opt in ((good_p1, good_o1), (good_p2, good_o2)):
        torch.testing.assert_close(p, reference_p, rtol=0.0, atol=0.0)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                opt.state[p][name],
                reference_o.state[reference_p][name],
                rtol=0.0, atol=0.0,
            )
    for name, arr in buffers.items():
        np.testing.assert_array_equal(arr, original[name])
    print("GREEN: independent restores, immutable sender and exact "
          "next-step Adam state in both receivers")
    print("CPU mechanism only: torch", torch.__version__,
          "numpy", np.__version__, "no Ray actor or native Ray CI")


if __name__ == "__main__":
    main()
