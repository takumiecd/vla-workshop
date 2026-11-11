"""Verify the LIBERO/data runtime without depending on a workshop package."""

import argparse
import json
from importlib.metadata import version


def check(require_cuda=True):
    import av
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    expected = {
        "numpy": "2.1.3",
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "lerobot": "0.7.0",
        "mujoco": "3.3.2",
        "robosuite": "1.4.0",
        "hf-libero": "0.1.4",
    }
    installed = {name: version(name) for name in expected}
    if installed != expected:
        raise RuntimeError(f"Unexpected versions: {installed}")
    np.testing.assert_array_equal(torch.from_numpy(np.ones(3)).numpy(), np.ones(3))
    assert callable(LeRobotDataset.create) and callable(av.open)
    result = {"packages": installed, "cuda_tested": require_cuda}
    if require_cuda:
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("T4 GPUを選んでください")
        assert torch.ones(3, device="cuda").sum().item() == 3
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = json.dumps(check(not args.no_cuda), indent=2) + "\n"
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(result)
    print(result)
