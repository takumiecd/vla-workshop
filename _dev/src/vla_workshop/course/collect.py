"""Collect policy rollouts for review, then export explicitly accepted episodes."""

import importlib.util
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from vla_workshop.config import PolicyConfig
from vla_workshop.envs.robosuite_env import state_vector, validate_action
from vla_workshop.policies.smolvla import SmolVLA

from .common import output_directory, read_json, write_json
from .data import IMAGE_KEYS, write_episodes
from .libero import LiberoEnvironment, evaluation_cases


def load_adapter(path: str | None):
    """Load a participant-authored action correction; default is the model's action."""
    if path is None:
        return lambda observation, proposed, step: proposed
    if not Path(path).is_file() or Path(path).suffix != ".py":
        raise FileNotFoundError(f"Missing Python correction file: {path}")
    spec = importlib.util.spec_from_file_location("workshop_collection_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.correct_action


def collect(config: dict, adapter_path: str | None = None) -> Path:
    cases = evaluation_cases(config, "collection")
    policy = SmolVLA(PolicyConfig(**config["policy"]), 256)
    correct = load_adapter(adapter_path)
    folder = output_directory("collection")
    manifest = {
        "status": "running",
        "config": config,
        "episodes": [],
        "source": "policy rollouts with optional scripted corrections; not human demonstrations",
    }
    write_json(folder / "collection.json", manifest)
    try:
        for number, (task, initial) in enumerate(cases):
            seed = config["seed"] + task * 10000 + initial
            np.random.seed(seed)
            env = LiberoEnvironment(config, task, initial, seed)
            try:
                policy.reset(seed=seed)
                states, actions = [], []
                images = {key: [] for key in IMAGE_KEYS}
                episode_dir = folder / f"episode_{number:03d}"
                episode_dir.mkdir()
                with imageio.get_writer(episode_dir / "video.mp4", fps=20) as video:
                    video.append_data(env.obs["agentview_image"][::-1, ::-1])
                    for step in range(config["max_steps"]):
                        proposed = policy.select_action(env.obs, env.instruction)
                        action = validate_action(
                            correct(env.obs, proposed.copy(), step), env.low, env.high
                        )
                        states.append(state_vector(env.obs))
                        actions.append(action)
                        for key, camera in zip(
                            IMAGE_KEYS, ["agentview_image", "robot0_eye_in_hand_image"], strict=True
                        ):
                            images[key].append(np.ascontiguousarray(env.obs[camera][::-1, ::-1]))
                        _, success, done = env.step(action)
                        video.append_data(env.obs["agentview_image"][::-1, ::-1])
                        if success or done:
                            break
                np.savez_compressed(
                    episode_dir / "frames.npz",
                    states=np.stack(states),
                    actions=np.stack(actions),
                    **{k: np.stack(v) for k, v in images.items()},
                )
                row = {
                    "episode": number,
                    "task_id": task,
                    "initial_state": initial,
                    "seed": seed,
                    "steps": len(actions),
                    "success": success,
                    "instruction": env.instruction,
                    "initial_state_sha256": env.initial_state_hash,
                    "adapter": Path(adapter_path).read_text() if adapter_path else None,
                }
                write_json(episode_dir / "episode.json", row)
                manifest["episodes"].append(row)
                write_json(folder / "collection.json", manifest)
            finally:
                env.close()
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(folder / "collection.json", manifest)
    print(f"Results: {folder}", flush=True)
    return folder


def export_accepted(folder: str | Path, accepted: list[int], destination: str | Path) -> Path:
    folder, destination = Path(folder), Path(destination)
    report = read_json(folder / "collection.json")
    if report["status"] != "completed":
        raise ValueError("Review a completed collection before exporting")
    if (
        not accepted
        or len(set(accepted)) != len(accepted)
        or any(type(index) is not int or index < 0 for index in accepted)
    ):
        raise ValueError("Select at least one unique accepted episode")
    available = {row["episode"]: row for row in report["episodes"]}
    if not set(accepted).issubset(available):
        raise ValueError("Accepted episode does not exist")
    episodes = []
    for index in accepted:
        row = available[index]
        with np.load(folder / f"episode_{index:03d}" / "frames.npz", allow_pickle=False) as data:
            episodes.append(
                {
                    "states": data["states"],
                    "actions": data["actions"],
                    **{key: data[key] for key in IMAGE_KEYS},
                    "instruction": row["instruction"],
                    "provenance": {"collection": str(folder), **row},
                }
            )
    write_episodes(destination, episodes)
    print(f"Results: {destination}", flush=True)
    return destination
