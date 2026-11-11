"""Small, explicit JSON configuration shared by the lesson entry points."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnvConfig:
    task: str = "Lift"
    robot: str = "Panda"
    cameras: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    image_size: int = 256
    control_freq: int = 20
    settle_steps: int = 10
    renderer: str = "egl"


@dataclass
class PolicyConfig:
    path: str
    revision: str | None = None
    vlm_revision: str | None = None
    device: str = "cuda"
    n_action_steps: int = 1
    camera_map: dict[str, str] = field(default_factory=dict)
    image_transform: str = "rotate180"


@dataclass
class RunConfig:
    env: EnvConfig
    policy: PolicyConfig
    seed: int = 0
    episodes: int = 1
    max_steps: int = 200
    output_dir: str = "outputs"
    instruction: str = "pick up the cube"


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    config = RunConfig(
        env=EnvConfig(**data.pop("env")),
        policy=PolicyConfig(**data.pop("policy")),
        **data,
    )
    if config.env.task != "Lift" or config.env.robot != "Panda":
        raise ValueError("This first lesson supports Lift + Panda with OSC_POSE only.")
    if (
        config.episodes < 1
        or config.max_steps < 1
        or config.env.image_size < 16
        or config.env.control_freq < 1
        or config.env.settle_steps < 0
    ):
        raise ValueError("Invalid episode, step, camera size, or control frequency setting.")
    if not config.instruction.strip():
        raise ValueError("instruction must not be empty")
    if (
        not config.env.cameras
        or len(set(config.env.cameras)) != len(config.env.cameras)
        or not set(config.policy.camera_map.values()).issubset(config.env.cameras)
    ):
        raise ValueError("camera_map must refer to configured, unique cameras")
    if config.policy.n_action_steps < 1:
        raise ValueError("n_action_steps must be positive")
    if config.policy.image_transform not in {"rotate180", "flip_vertical", "identity"}:
        raise ValueError("Unknown image_transform")
    if config.env.renderer not in {"egl", "osmesa", "glfw"}:
        raise ValueError("renderer must be egl, osmesa, or glfw")
    # Relative outputs and local checkpoints are relative to the project, not the caller's cwd.
    project = path.parent.parent if path.parent.name == "configs" else path.parent
    config.output_dir = str((project / config.output_dir).resolve())
    local_path = (project / config.policy.path).resolve()
    if local_path.is_dir():
        config.policy.path = str(local_path)
    return config
