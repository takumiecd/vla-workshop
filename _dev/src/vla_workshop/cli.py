"""Installed commands and the numbered lessons share these entry points."""

import argparse
from pathlib import Path

from vla_workshop.config import RunConfig, load_config


def _config(description: str) -> RunConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    for name in ("episodes", "max_steps"):
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                parser.error(f"{name} must be positive")
            setattr(config, name, value)
    return config


def inspect_main() -> None:
    config = _config("Inspect standard robosuite Lift/Panda observations and controls")
    from vla_workshop.evaluation.rollout import HoldPolicy, run

    run(config, HoldPolicy(), "inspect")


def rollout_main() -> None:
    config = _config("Run SmolVLA on standard robosuite Lift/Panda using CUDA")
    # Also seed policy construction; per-episode sampling seeds are set in run().
    import torch

    from vla_workshop.evaluation.rollout import run
    from vla_workshop.policies.smolvla import SmolVLA

    torch.manual_seed(config.seed)
    policy = SmolVLA(config.policy, config.env.image_size)
    run(config, policy, "smolvla")
