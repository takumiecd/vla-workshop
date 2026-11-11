"""SmolVLA inference with the checkpoint's own normalizer and unnormalizer."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from config import PolicyConfig
from robosuite_env import state_vector

if TYPE_CHECKING:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def transform_image(image: np.ndarray, transform: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected HWC uint8 RGB image")
    if transform == "rotate180":
        image = image[::-1, ::-1]
    elif transform == "flip_vertical":
        image = image[::-1]
    elif transform != "identity":
        raise ValueError(f"Unknown image transform: {transform}")
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0


def validate_features(
    policy_config: "SmolVLAConfig",
    camera_map: dict[str, str],
    image_size: int,
) -> None:
    expected = policy_config.input_features
    actual = {"observation.state": (8,)}
    actual.update({key: (3, image_size, image_size) for key in camera_map})
    for key, feature in expected.items():
        if actual.get(key) != tuple(feature.shape):
            raise ValueError(
                f"Checkpoint feature mismatch: {key} expects {feature.shape}, "
                f"adapter provides {actual.get(key)}. Use a compatible checkpoint/camera_map."
            )
    if set(expected) != set(actual):
        raise ValueError("camera_map contains features absent from the checkpoint")
    if tuple(policy_config.output_features["action"].shape) != (7,):
        raise ValueError("This adapter requires a checkpoint with 7D OSC_POSE actions")


class SmolVLA:
    def __init__(self, config: PolicyConfig, image_size: int) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.policies.factory import make_pre_post_processors

        # Import registers the checkpoint's newline processor with LeRobot.
        from lerobot.policies.smolvla import processor_smolvla  # noqa: F401
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if not config.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Install CUDA PyTorch and check nvidia-smi.")
        self.torch = torch
        self.config = config
        local = Path(config.path).is_dir()
        if not local and not config.revision:
            raise ValueError(
                "Set policy.revision to a checkpoint commit for reproducible inference"
            )
        checkpoint = (
            config.path
            if local
            else snapshot_download(
                config.path,
                revision=config.revision,
                allow_patterns=["*.json", "*.safetensors"],
            )
        )
        policy_config = SmolVLAConfig.from_pretrained(checkpoint)
        validate_features(policy_config, config.camera_map, image_size)
        policy_config.device = config.device
        policy_config.load_vlm_weights = False  # Complete policy weights are in the checkpoint.
        if not 1 <= config.n_action_steps <= policy_config.chunk_size:
            raise ValueError("n_action_steps must be between 1 and checkpoint chunk_size")
        policy_config.n_action_steps = config.n_action_steps
        vlm_repo = policy_config.vlm_model_name
        if not config.vlm_revision:
            raise ValueError("Set policy.vlm_revision for the checkpoint's VLM tokenizer/config")
        vlm_path = snapshot_download(
            vlm_repo,
            revision=config.vlm_revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
        )
        policy_config.vlm_model_name = vlm_path
        self.policy = SmolVLAPolicy.from_pretrained(
            checkpoint,
            config=policy_config,
            strict=True,
        ).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": config.device},
                "tokenizer_processor": {"tokenizer_name": vlm_path},
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.metadata = {
            "checkpoint": checkpoint,
            "vlm_repo": vlm_repo,
            "vlm_snapshot": vlm_path,
            "device": str(next(self.policy.parameters()).device),
            "gpu": torch.cuda.get_device_name(torch.device(config.device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": str(next(self.policy.parameters()).dtype),
            "n_action_steps": config.n_action_steps,
            "num_steps": policy_config.num_steps,
        }

    def reset(self, *, seed: int) -> None:
        self.torch.manual_seed(seed)
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def select_action(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(state_vector(obs)), "task": instruction}
        for key, camera in self.config.camera_map.items():
            batch[key] = torch.from_numpy(
                transform_image(
                    obs[f"{camera}_image"],
                    self.config.image_transform,
                )
            )
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        action = action.squeeze(0).detach().cpu().numpy()
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("SmolVLA returned an invalid 7D action")
        return action
