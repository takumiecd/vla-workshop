"""Check the locked numerical, simulation, and SmolVLA imports before a lesson."""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path


def check_environment(require_cuda: bool = True) -> dict[str, object]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    import robosuite
    import torch
    import torchvision
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    expected = {
        "numpy": "2.1.3",
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "mujoco": "3.3.2",
        "lerobot": "0.7.0",
        "transformers": "5.22.0",
    }
    installed = {name: version(name) for name in expected}
    for name, target in expected.items():
        if installed[name] != target:
            raise RuntimeError(f"{name}: expected {target}, installed {installed[name]}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA 12.8 build, got {torch.version.cuda}")
    # Exercise NumPy/Torch conversion, not just module import.
    np.testing.assert_array_equal(
        torch.from_numpy(np.ones(3, dtype=np.float32)).numpy(), np.ones(3)
    )
    assert SmolVLAConfig is not None and SmolVLAPolicy is not None
    assert callable(make_pre_post_processors) and callable(mujoco.mj_versionString)
    assert callable(robosuite.make) and callable(torchvision.transforms.functional.to_tensor)
    result = {
        "python": sys.version,
        "packages": installed,
        "cuda_build": torch.version.cuda,
        "cuda_tested": require_cuda,
    }
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAが使えません。ColabのT4 GPUに接続してください。")
        gpu = torch.cuda.get_device_name(0)
        if "T4" not in gpu.upper():
            raise RuntimeError(f"T4 GPUを選択してください。現在のGPU: {gpu}")
        if "sm_75" not in torch.cuda.get_arch_list():
            raise RuntimeError("このPyTorchビルドにはT4用のsm_75が含まれていません。")
        assert torch.ones(16, device="cuda").sum().item() == 16
        result.update(gpu=gpu, capability=list(torch.cuda.get_device_capability(0)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-cuda", action="store_true", help="CPU import checks only")
    args = parser.parse_args()
    result = check_environment(require_cuda=not args.no_cuda)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)
