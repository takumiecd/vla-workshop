"""Read, inspect and export episode-disjoint LeRobot datasets."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .common import digest, episode_split, output_directory, write_json

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def open_dataset(config: dict, episodes: list[int], action_times=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = config.get("root")
    if root and not (Path(root) / "meta/info.json").exists():
        raise FileNotFoundError(f"Local dataset is missing meta/info.json: {root}")
    if not root and len(config.get("revision", "")) != 40:
        raise ValueError("Pin the Hub dataset revision to a commit")
    dataset = LeRobotDataset(
        config["repo_id"],
        root=Path(root) if root else None,
        episodes=episodes,
        revision=config.get("revision"),
        video_backend="pyav",
        delta_timestamps={"action": action_times} if action_times is not None else None,
    )
    features = dataset.meta.features
    for key, shape in [("observation.state", (8,)), ("action", (7,))]:
        if tuple(features[key]["shape"]) != shape:
            raise ValueError(f"Incompatible dataset feature: {key}")
    for key in IMAGE_KEYS:
        if tuple(features[key]["shape"]) != (256, 256, 3):
            raise ValueError(f"Expected 256x256 RGB dataset camera: {key}")
    if len(dataset) < 1:
        raise ValueError("Selected dataset is empty")
    return dataset


def inspect_dataset(config: dict) -> Path:
    train, validation = episode_split(config)
    dataset = open_dataset(config, train + validation)
    folder = output_directory("dataset")
    selected = set(train + validation)
    counts = {str(index): 0 for index in selected}
    action_min = np.full(7, np.inf)
    action_max = np.full(7, -np.inf)
    instructions = {str(index): set() for index in selected}
    task_names = {int(row["task_index"]): str(task) for task, row in dataset.meta.tasks.iterrows()}
    # Scan numeric columns only. This avoids decoding every camera frame.
    for row in dataset.hf_dataset.select_columns(
        ["episode_index", "task_index", "action", "observation.state"]
    ):
        episode = int(row["episode_index"])
        if episode not in selected:
            continue
        action, state = np.asarray(row["action"]), np.asarray(row["observation.state"])
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError(f"Non-finite data in episode {episode}")
        counts[str(episode)] += 1
        instructions[str(episode)].add(task_names[int(row["task_index"])])
        action_min, action_max = np.minimum(action_min, action), np.maximum(action_max, action)
    if any(count == 0 for count in counts.values()):
        raise ValueError("Some requested episodes have no frames")
    replay_episode = config.get("replay_episode", train[0])
    if replay_episode not in selected:
        raise ValueError("Replay episode must belong to the inspected split")
    replay = open_dataset(config, [replay_episode])
    with imageio.get_writer(folder / "replay.mp4", fps=dataset.fps) as video:
        for index in range(min(len(replay), config.get("preview_frames", 100))):
            frame = replay[index]
            image = np.concatenate(
                [frame[key].permute(1, 2, 0).numpy() for key in IMAGE_KEYS], axis=1
            )
            video.append_data(np.clip(image * 255, 0, 255).astype(np.uint8))
    write_json(
        folder / "dataset.json",
        {
            "status": "completed",
            "config": config,
            "config_sha256": digest(config),
            "frames_per_episode": counts,
            "fps": dataset.fps,
            "instructions": {key: sorted(value) for key, value in instructions.items()},
            "features": dataset.meta.features,
            "action_min": action_min.tolist(),
            "action_max": action_max.tolist(),
            "preview_episode": replay_episode,
            "note": "Episode-disjoint workshop split; checkpoint pretraining overlap is unknown",
        },
    )
    print(f"Results: {folder}", flush=True)
    return folder


def write_episodes(root: Path, episodes: list[dict]) -> dict:
    """Create an image-backed LeRobot v3 dataset with explicit collection provenance."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        raise FileExistsError(f"Choose a new dataset directory: {root}")
    if not episodes or any(len(ep["actions"]) == 0 for ep in episodes):
        raise ValueError("No nonempty accepted episodes")
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    features.update(
        {
            key: {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            }
            for key in IMAGE_KEYS
        }
    )
    dataset = LeRobotDataset.create(
        repo_id="workshop/collected_libero",
        fps=20,
        features=features,
        root=root,
        robot_type="libero_panda",
        video_backend="pyav",
        use_videos=False,
    )
    provenance = []
    try:
        for index, episode in enumerate(episodes):
            n = len(episode["actions"])
            if len(episode["states"]) != n or any(len(episode[k]) != n for k in IMAGE_KEYS):
                raise ValueError("Observation/action lengths disagree")
            for frame in range(n):
                dataset.add_frame(
                    {
                        "observation.state": np.asarray(episode["states"][frame], dtype=np.float32),
                        "action": np.asarray(episode["actions"][frame], dtype=np.float32),
                        "task": episode["instruction"],
                        **{key: episode[key][frame] for key in IMAGE_KEYS},
                    }
                )
            dataset.save_episode()
            provenance.append({"episode_index": index, **episode["provenance"]})
    finally:
        dataset.finalize()
    manifest = {
        "status": "completed",
        "episodes": provenance,
        "image_orientation": "rotate180 from raw LIBERO observations",
        "fps": 20,
    }
    write_json(root / "workshop_provenance.json", manifest)
    return manifest
