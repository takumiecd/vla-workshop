"""Load episode-disjoint data and save the trained model with its processors."""

from pathlib import Path

from common import episode_split

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def open_dataset(config: dict, episodes: list[int], action_times=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if len(config.get("revision", "")) != 40:
        raise ValueError("Pin the Hub dataset revision to a commit")
    # Keep the selected episode files in their own cache. LeRobot can otherwise
    # see only the shards downloaded for a different subset of the same dataset.
    selection = "-".join(map(str, episodes))
    root = Path("datasets") / config["repo_id"] / config["revision"] / selection
    dataset = LeRobotDataset(
        config["repo_id"],
        root=root,
        episodes=episodes,
        revision=config.get("revision"),
        video_backend="pyav",
        delta_timestamps={"action": action_times} if action_times is not None else None,
    )
    features = dataset.features
    for key, shape in [("observation.state", (8,)), ("action", (7,))]:
        if tuple(features[key]["shape"]) != shape:
            raise ValueError(f"Incompatible dataset feature: {key}")
    for key in IMAGE_KEYS:
        if tuple(features[key]["shape"]) != (256, 256, 3):
            raise ValueError(f"Expected 256x256 RGB dataset camera: {key}")
    if len(dataset) < 1:
        raise ValueError("Selected dataset is empty")
    return dataset


def load_training_data(config, policy):
    train_ids, validation_ids = episode_split(config)
    # LeRobot interprets these offsets in seconds, using the dataset's recorded fps.
    first_episode = open_dataset(config, train_ids[:1])
    offsets = [index / first_episode.fps for index in policy.config.action_delta_indices]
    training = open_dataset(config, train_ids, offsets)
    validation = open_dataset(config, validation_ids, offsets)
    return training, validation


def save_model(policy, pre, post, vlm_repo, folder: Path):
    folder.mkdir(parents=True)
    # Keep the tokenizer/config portable after the Colab runtime is deleted.
    config = policy.config
    local_vlm = config.vlm_model_name
    try:
        config.vlm_model_name = vlm_repo
        policy.save_pretrained(folder)
    finally:
        config.vlm_model_name = local_vlm
    pre.save_pretrained(folder, config_filename="policy_preprocessor.json")
    post.save_pretrained(folder, config_filename="policy_postprocessor.json")
