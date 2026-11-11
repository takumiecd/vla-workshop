"""Prepare pinned model files and convert a robot observation to LeRobot inputs."""

from pathlib import Path

import numpy as np
from lift_tools import state_vector


def checkpoint_inputs(settings: dict, image_size: int):
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    checkpoint = settings["path"]
    if not Path(checkpoint).is_dir():
        if not settings.get("revision"):
            raise ValueError("Specify the Hub checkpoint revision")
        checkpoint = snapshot_download(
            checkpoint, revision=settings["revision"], allow_patterns=["*.json", "*.safetensors"]
        )
    config = SmolVLAConfig.from_pretrained(checkpoint)
    shapes = {"observation.state": (8,)}
    shapes.update({key: (3, image_size, image_size) for key in settings["camera_map"]})
    if shapes != {key: tuple(feature.shape) for key, feature in config.input_features.items()}:
        raise ValueError("Camera or state shapes do not match this checkpoint")
    if tuple(config.output_features["action"].shape) != (7,):
        raise ValueError("This exercise requires a 7D robot action")
    config.device = settings["device"]
    config.load_vlm_weights = False
    config.n_action_steps = settings["n_action_steps"]
    if not 1 <= config.n_action_steps <= config.chunk_size:
        raise ValueError("n_action_steps must be between 1 and chunk_size")
    vlm_repo = config.vlm_model_name
    config.vlm_model_name = snapshot_download(
        vlm_repo,
        revision=settings["vlm_revision"],
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
    )
    return checkpoint, config, vlm_repo


def observation_batch(obs: dict, instruction: str, settings: dict) -> dict:
    import torch

    batch = {"observation.state": torch.from_numpy(state_vector(obs)), "task": instruction}
    for key, camera in settings["camera_map"].items():
        image = obs[f"{camera}_image"]
        transform = settings["image_transform"]
        if transform == "rotate180":
            image = image[::-1, ::-1]
        elif transform == "flip_vertical":
            image = image[::-1]
        elif transform != "identity":
            raise ValueError(f"Unknown image transform: {transform}")
        chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        batch[key] = torch.from_numpy(chw / 255.0)
    return batch
